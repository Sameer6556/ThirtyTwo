# Accuracy check

The deployed system (`thirtytwo.pipeline`, exactly as the app runs it, one X-ray at a time on the PNG bytes) on protocol A of the v15 notebook, scored with pycocotools (COCO AP over IoU 0.50 to 0.95, all boxes) by the notebook's own `dentex_utils.dentex_eval`. CPU only, fp32. Reproduce with `python tools/accuracy_check.py`.

## Official 50 DENTEX validation X-rays

| | AP | AP50 | AP75 | report AP | report AP50 | check |
|---|---|---|---|---|---|---|
| Full system | 48.77 | 69.49 | 58.62 | 48.77 | 69.40 | pass (target 48.8 / 69.4 +-0.3) |
| Quick mode | 48.15 | 69.57 | 58.15 | 48.08 | 69.49 | pass (target 48.1 +-0.3) |
| Saved Kaggle outputs, fused by the app's fusion code | 48.77 | 69.40 | 58.53 | | | |

Per disease (full system, AP): Caries 41.4, Deep Caries 55.8, Periapical Lesion 37.8, Impacted 60.1.

On the 70 protocol-A validation X-rays: full 48.55 AP (report 48.69), Quick 44.39 AP (report 44.40).

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

50 of 50 X-rays have the same tooth numbers (0 bit for bit). Over 1328 matched teeth: lowest IoU 0.999991, largest box edge difference 4.9e-04 px, largest confidence difference 4.2e-06 (GPU vs CPU floating point).

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

### Tooth detector vs teeth_clean.pkl (70 A_val)

70 of 70 X-rays have the same tooth numbers (0 bit for bit). Over 1937 matched teeth: lowest IoU 0.999992, largest box edge difference 7.3e-04 px, largest confidence difference 5.2e-06 (GPU vs CPU floating point).

### Classifiers vs cls/*_E/pred_*.csv (70 A_val)

| classifier | teeth | median abs diff | 99th percentile | max |
|---|---|---|---|---|
| crop only | 1937 | 2.3e-06 | 1.6e-03 | 0.0075 |
| self-trained crop only | 1937 | 1.7e-06 | 1.3e-03 | 0.0096 |

### YOLOs vs disbox_*_validator.pkl (70 A_val)

| YOLO | identical X-rays | boxes | count diff | median IoU | boxes IoU < 0.9 | of them with conf >= 0.1 | median conf diff | max conf diff |
|---|---|---|---|---|---|---|---|---|
| plain YOLO | 0 / 70 | 6298 | 33 | 1.0 | 77 | 0 of 825 | 0.0e+00 | 0.703 |
| plain YOLO, seed 1 | 0 / 70 | 4022 | 14 | 1.0 | 30 | 1 of 720 | 0.0e+00 | 0.7514 |
| plain YOLO at 1280 | 11 / 70 | 2240 | 18 | 1.0 | 26 | 1 of 503 | 0.0e+00 | 0.8628 |
| round-1 self-trained YOLO11m | 18 / 70 | 2694 | 13 | 1.0 | 12 | 0 of 498 | 0.0e+00 | 0.5292 |
| round-2 self-trained YOLO11m | 12 / 70 | 2817 | 12 | 1.0 | 6 | 0 of 586 | 0.0e+00 | 0.8442 |
| round-2 self-trained YOLO26m | 11 / 70 | 2529 | 14 | 1.0 | 11 | 0 of 563 | 0.0e+00 | 0.8664 |
| round-3 self-trained YOLO26m | 4 / 70 | 2488 | 31 | 1.0 | 59 | 0 of 472 | 0.0e+00 | 0.8604 |

## Display thresholds (70 protocol-A validation X-rays, best F1 at IoU 0.5)

| disease | full | F1 | quick | F1 |
|---|---|---|---|---|
| Caries | 0.3571 | 0.567 | 0.2496 | 0.559 |
| Deep caries | 0.2110 | 0.634 | 0.2581 | 0.610 |
| Periapical lesion | 0.1064 | 0.579 | 0.1208 | 0.514 |
| Impacted tooth | 0.1900 | 0.863 | 0.3448 | 0.851 |

## Time per X-ray on this machine (4 CPU cores)

Full 10.5 s on average (slowest 13.39 s), Quick 1.15 s (slowest 1.45 s), models already loaded.

