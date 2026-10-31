# Accuracy check

The deployed system (`thirtytwo.pipeline`, exactly as the app runs it, one X-ray at a time on the PNG bytes) on protocol A of the v15 notebook, scored with pycocotools (COCO AP over IoU 0.50 to 0.95, all boxes) by the notebook's own `dentex_utils.dentex_eval`. CPU only, fp32. Reproduce with `python tools/accuracy_check.py`.

## Official 50 DENTEX validation X-rays

| | AP | AP50 | AP75 | report AP | report AP50 | check |
|---|---|---|---|---|---|---|
| Full system | 48.77 | 69.49 | 58.62 | 48.77 | 69.40 | pass (target 48.8 / 69.4 +-0.3) |
| Quick mode | 48.15 | 69.57 | 58.15 | 48.08 | 69.49 | pass (target 48.1 +-0.3) |
| Saved Kaggle outputs, fused by the app's fusion code | 48.77 | 69.40 | 58.53 | | | |

Per disease (full system, AP): Caries 41.4, Deep Caries 55.8, Periapical Lesion 37.8, Impacted 60.1.

## Every member against the saved outputs

### AP of each member alone (official 50)

| member | AP | AP50 | report AP | report AP50 | difference |
|---|---|---|---|---|---|
| crop only | 32.21 | 52.41 | 32.21 | 52.41 | +0.00 |
| self-trained crop only | 34.92 | 56.65 | 34.91 | 56.63 | +0.01 |
| plain YOLO | 37.86 | 55.64 | 37.97 | 55.68 | -0.11 |
| plain YOLO, seed 1 | 40.55 | 59.92 | 40.52 | 59.97 | +0.03 |
| plain YOLO at 1280 | 39.76 | 58.49 | 39.92 | 58.48 | -0.16 |
| round-1 self-trained YOLO11m | 48.15 | 69.57 | 48.08 | 69.49 | +0.07 |
| round-2 self-trained YOLO11m | 46.64 | 67.63 | 46.63 | 67.65 | +0.01 |
| round-2 self-trained YOLO26m | 46.72 | 66.62 | 46.74 | 66.69 | -0.02 |
| round-3 self-trained YOLO26m | 48.13 | 67.02 | 48.03 | 66.97 | +0.10 |

### Tooth detector vs teeth_clean.pkl (official 50)

0 of 50 X-rays identical; 50 with the same tooth numbers. Over 1328 matched teeth: median IoU 1.0, lowest 1.0, largest box edge difference 0.0 px, largest confidence difference 0.0.

### Classifiers vs cls/*_E/pred_*.csv (official 50)

| classifier | teeth | median abs diff | 99th percentile | max |
|---|---|---|---|---|
| crop only | 1328 | 2.4e-06 | 1.6e-03 | 0.0098 |
| self-trained crop only | 1328 | 1.8e-06 | 1.6e-03 | 0.0077 |

### YOLOs vs disbox_*_validator.pkl (official 50)

| YOLO | identical X-rays | boxes | count diff | median IoU | boxes IoU < 0.9 | of them with conf >= 0.1 | median conf diff | max conf diff |
|---|---|---|---|---|---|---|---|---|
| plain YOLO | 0 / 50 | 4377 | 67 | 1.0 | 123 | 1 of 578 | 0.0e+00 | 0.7973 |
| plain YOLO, seed 1 | 1 / 50 | 2828 | 30 | 1.0 | 51 | 1 of 465 | 0.0e+00 | 0.7831 |
| plain YOLO at 1280 | 3 / 50 | 1631 | 30 | 1.0 | 29 | 0 of 359 | 0.0e+00 | 0.8306 |
| round-1 self-trained YOLO11m | 9 / 50 | 1889 | 4 | 1.0 | 3 | 0 of 327 | 0.0e+00 | 0.7822 |
| round-2 self-trained YOLO11m | 8 / 50 | 1927 | 8 | 1.0 | 10 | 0 of 358 | 0.0e+00 | 0.0345 |
| round-2 self-trained YOLO26m | 12 / 50 | 1794 | 5 | 1.0 | 6 | 0 of 353 | 0.0e+00 | 0.072 |
| round-3 self-trained YOLO26m | 6 / 50 | 1714 | 107 | 1.0 | 117 | 0 of 323 | 0.0e+00 | 0.8623 |

## Time per X-ray on this machine (4 CPU cores)

Full 10.5 s on average (slowest 13.39 s), Quick 1.15 s (slowest 1.45 s), models already loaded.

