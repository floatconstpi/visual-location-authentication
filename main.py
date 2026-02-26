from utils import *
from pdb import set_trace as st
from matplotlib import pyplot as plt
import numpy as np
import argparse
from torch.utils.data import DataLoader
from sklearn import metrics
import pandas as pd
import csv

def pinhole_range_est_exp(dataset,max_dist=60,inc=10,predicted=False):
    """
    ## Pinhole Range Estimation RMSE. Total average, and per distance bin.
    """
    err_abs = []
    RMSE_per_bin = {key+inc: [] for key in range(0,max_dist,inc)}
    for sample in dataset:
        img, distance, ann, full_ann, scene, name = sample
        bb = ann['bbox']   
        bin = int((distance//inc)*inc+inc)

        pinhole_dist = pinhole_distance_estimate(bb, scene)
        err_abs.append(abs(pinhole_dist-distance))

        RMSE_per_bin[bin].append(abs(pinhole_dist-distance))
        
        
    print("Total RMSE: ",np.sqrt(np.mean(err_abs)))
    for key, value in RMSE_per_bin.items():
        if len(value) == 0: continue
        print(f"RMSE for bin {str(key)} is {np.mean(value)}\n")

def distance_spoof_exp(dataset,taus=None,show_ROC=True,seed=None,data_name=None, eps_mags=None, exp_name=None):
    labels = ["authentic","non-authentic"]
    if taus is None:
        taus = np.arange(0.0, 60.0 + 1e-9, 0.2)

    all_flag_auth = []
    all_flag_spoof = []
    all_eps_mag = []
    distances = []
    all_results = []
    # save_dir = f"seed_{seed}"+f"_exp_{exp_name}"*(exp_name is not None)
    save_dir = f"seed_{seed}_{eps_mags[0]}_{eps_mags[1]}"
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    for idx, (img, auth_distance, ann, full_ann, scene, name) in enumerate(dataset):
        spoof_dist, eps, eps_mag = make_rid_injection(auth_distance,eps_mags=eps_mags,seed=seed+idx)
        spoof_dist = float(spoof_dist)
        eps_mag = float(eps_mag)

        bb = ann["bbox"]
        z_vis = pinhole_distance_estimate(bb, scene)  # <- your function

        flag_auth, flag_spoof = decision_making(auth_distance, spoof_dist, z_vis, taus)
        all_flag_auth.append(flag_auth)
        all_flag_spoof.append(flag_spoof)
        all_eps_mag.append(eps_mag)
        distances.append([auth_distance,spoof_dist,z_vis,abs(z_vis-auth_distance),abs(z_vis-spoof_dist)])
    
    distances = pd.DataFrame(distances,columns=['Authentic Distance','Spoofed Distance','Perceived Distance','diff to authentic','diff to non-authentic' ])    
    
    all_flag_auth = np.stack(all_flag_auth, axis=0)    # (N, T)
    all_flag_spoof = np.stack(all_flag_spoof, axis=0)  # (N, T)
    

    fpr = all_flag_auth.mean(axis=0)  # P(flag | adherent)
    tpr = all_flag_spoof.mean(axis=0) # P(flag | non-adherent)

    auc_roc = metrics.auc(fpr, tpr)
    print("="*15,data_name,"="*15)
    print("AUC =", auc_roc)
    

    if show_ROC:
        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0, 1], [0, 1], "--", linewidth=1)  # chance line
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC: distance inconsistency flagging")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/Fig_{data_name}.png")

    chosen_tau_low_fpr,_,_ = select_tau_low_fpr(taus, tpr, fpr)
    chosen_tau_youden,_,_,_ = select_tau_youden(taus, tpr, fpr)
    tau_idx = list(taus).index(chosen_tau_low_fpr)
    print(f"Thresold to get lowest FPR: {chosen_tau_low_fpr}")
    print(f"TPR: {tpr[tau_idx]}")
    print(f"FPR: {fpr[tau_idx]}")
    all_results.extend([f"{tpr[tau_idx]:.3f}",f"{fpr[tau_idx]:.3f}"])
    distances['flagged authnetic (1) as'] = [labels[int(i)] for i in all_flag_auth[:,tau_idx]]
    distances['Flagged non-authentic (1) as'] = [labels[int(i)] for i in all_flag_spoof[:,tau_idx]]
    
    print("."*50)
    print(f"Thresold according to Youden's J: {chosen_tau_youden}")
    tau_idx = list(taus).index(chosen_tau_youden)
    print(f"TPR: {tpr[tau_idx]}")
    print(f"FPR: {fpr[tau_idx]}")
    all_results.extend([f"{tpr[tau_idx]:.3f}",f"{fpr[tau_idx]:.3f}"])
    distances['flagged authnetic (2) as'] = [labels[int(i)] for i in all_flag_auth[:,tau_idx]]
    distances['Flagged non-authentic (2) as'] = [labels[int(i)] for i in all_flag_spoof[:,tau_idx]]
    print("="*50)
    distances.to_csv(f"{save_dir}/distances_{data_name}.csv", index=False)
    all_results.extend([f"{auc_roc:.3f}"])
    with open(f"{save_dir}/results.csv","a") as f:
        writer = csv.writer(f)
        writer.writerow(["TPR@1","FPR@1","TPR@2","FPR@2","AUC"])
        writer.writerow(all_results)
        writer.writerow(["chosen_tau_low_fpr","chosen_tau_youden"])
        writer.writerow([chosen_tau_low_fpr,chosen_tau_youden])
    

    return taus, tpr, fpr, np.array(all_eps_mag)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Location Authentication Experimentation")
    parser.add_argument("--exp", help="This argument decides which experiment to run", choices=("range","spoof","det"))
    parser.add_argument("--data", help="Dataset to test", choices=("univ_1","univ_2","indoor_1","indoor_2","farm"))
    parser.add_argument("--eps_mags", nargs='+',help="space-separated eps magnitudes to test with",type=float,default=[2,5,10])
    parser.add_argument("--exp_name", help="additional experiment identifier",type=str,default="trial")
    parser.add_argument("--seed", help="random seed", type=int)
    parser.add_argument("--predicted", help="to use predicted bbox vs GT", action='store_true')
    args = parser.parse_args()

    dataset = DroneDataset("../drones_datasets/DroneR",args.data,predicted=args.predicted)


    if args.exp == "range":
        pinhole_range_est_exp(dataset,predicted=args.predicted)
    elif args.exp == "spoof":
        distance_spoof_exp(dataset,seed=args.seed,data_name=args.data, eps_mags=args.eps_mags, exp_name=args.exp_name)