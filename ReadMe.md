UTM - Visual Authentication of Drone Location  
===========

***Overview:** Our tool represents a visual authentication layer of drone-broadcasted RID. The tool addresses a serious problem in current drone RID broadcasting, where the authentication is not currently enforced under existing standards, making the broadcast RID easily spoofable. We confirm the received RID visually by comparing it to the perceived distance/location.
<div style="width: 100%; text-align: center;">
    <img src="Fig1-animated.gif" width="600px" />
</div>

[Installation](#installation) • [Dataset](#Dataset) • [Drone Detection](#drone-detection) • [Pinhole Range Estimation](#pinhole-range-estimation) • [Spoofing Detection](#spoofing-detection) • [Checkpoints](#checkpoints) •  [Cite](#reference)


## Installation:
Clone this repo
```
git clone https://github.com/KU-USL/visual-location-authentication.git
```

## Dataset
To use the same dataset DroneR featured in our paper, use the following link:
[Dataset Link](www.google.com)
The downloaded files will include a COCO format dataset, consisting of images, with manual bounding box annotations. Moreover, we supply the GT distances measured manually in Distances.csv along with other useful metadata.

Furthermore, we supply the precomputed bounding box predictions made by our [model](#Drone Detection)

## Drone Detection 
Run YOLO inference on your dataset to generate bounding box predictions in CSV format, compatible with the `--predicted` flag in main.py.
```
python detect.py --model best.pt --data data.yaml --output predictions.csv --imgsz 1152 --conf 0.25 --nms-iou 0.7 --device 0
```

- `--model`: Path to YOLO checkpoint (default: `best.pt`). 
- `--data`: YOLO format `data.yaml` defining dataset splits. 
- `--output`: CSV with filename, bbox_x/y/w/h, and IoU. 
- Adjust `--imgsz`, `--conf`, `--nms-iou`, `--device` as needed. Precomputed predictions are provided in the dataset for convenience.
## Pinhole Range Estimation
You can use the code directly to approximate the distance of a drone in a set of images. The code reports the average RMSE between the GT distance and the perceived distance. 
```
python main.py --exp range --data <dataset_name> --predicted
```
- The `--predicted` option fetches the dimensions of the drone from a csv file listing predictions made by a detection model. Otherwise, a json annotation is used.
- Outputs total RMSE and per-bin RMSE to console.

## Spoofing Detection 
Simulates RID spoofing attacks with epsilon magnitudes, evaluates flagging via distance inconsistency thresholds. Generates ROC plots, CSV results (distances, flags), and metrics (AUC, TPR/FPR at optimal taus).
```
python main.py --exp spoof --data univ_1 --eps_mags 20 50 --seed 999 --exp_name trial
```

- `--eps_mags`: Spoofing perturbation magnitudes (we use 20-50 and 2-6 for blatant attacks in the outdoor and indoor scenes, respectively. We use 7-20 and 0.5-2 for subtle attacks in the outdoor and indoor scenes, respectively). 
- `--seed`: For reproducible spoof distances. 
- `--exp_name`: Optional identifier for output dir. 
- Saves `seed_{seed}_{eps_min}_{eps_max}/` with `Fig_{data}.png`, `distances_{data}.csv`, `results.csv`. 
- Prints AUC, thresholds (low-FPR and Youden's J), TPR/FPR. Available datasets: `univ_1`, `univ_2`, `indoor_1`, `indoor_2`, `farm`.
## Checkpoints
For reproducibility, all trained models used can be accessed [here](www.google.com).

## Issues
- Please report all issues on the public forum.

## Reference
- Coming Soon!
