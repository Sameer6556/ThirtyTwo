# Thirty Two: record of the deployment work, for the project report

## How to use this file

This file records the engineering work that turned the research notebook of the project into a working, verified web
application called **Thirty Two**. It is written to be handed to an assistant (or a person) who writes the project
report. Every number in it was measured during the work, and the source of each is named (a file in the repository,
a test, or a tool). Two kinds of work are kept apart:

- **The research (done earlier, in the notebook).** Training the models, choosing their settings and evaluating them:
  the v15 notebook `dentex_twinarch_kaggle_2xT4.ipynb`, run on Kaggle (2x T4 GPUs). Its results are inputs here.
- **The deployment work (recorded here).** Rebuilding the trained system outside the notebook, proving that it gives
  the same results, and building, testing and packaging the web app around it.

Suggested report sections that can draw on this file: System implementation, Deployment, Verification and
reproducibility, Results (Accuracy), User interface design, Privacy, ethics and licensing, Testing, Limitations,
Future work.

Repository: https://github.com/Sameer6556/ThirtyTwo (branch `claude/lucid-davinci-mk3xmn`, pull request #1). All work
is dated 31 October 2026.

---

## 1. Summary

Thirty Two finds four problems on a panoramic dental X-ray (caries, deep caries, periapical lesions and impacted
teeth) and shows each finding as a box on the image with the FDI number of its tooth. It is named after the 32 adult
teeth. It runs the exact system of the notebook's protocol A (trained on 635 DENTEX X-rays, every setting picked on
70 held-out X-rays) on an ordinary CPU, inside a FastAPI web app with a plain HTML, CSS and JavaScript frontend.

Headline result of the verification: run one X-ray at a time on a CPU, the deployed app scores **48.77 AP and
69.49 AP50** on the 50 official DENTEX validation X-rays, against **48.77 / 69.40** in the notebook's report (target
48.8 / 69.4, tolerance 0.3). Quick mode scores **48.15 AP** (report 48.08).

---

## 2. Starting point

### 2.1 Inputs

| Input | What it is |
|---|---|
| `dentex_twinarch_kaggle_2xT4.ipynb` | The v15 notebook (RUN_TAG 'v15') that produced every number of the report. Saved without cell outputs. |
| `dentex_utils.py`, `train_arch.py` | The notebook's helper modules (data parsing, crops, fusion, evaluation; the classifier model). Checked to be identical to the notebook's `%%writefile` cells (`train_arch.py` differs by one trailing newline). They are imported, never rewritten. |
| `cusp_assets.zip` | Release `assets-v15` of the repository: 433 MB (432,676,162 bytes), 69 files, SHA-256 `2819760bf22bac3dfa7116c5cf0ed46c7c0b2d6800cdbf0a4c2da99c8515fc06`. Weights, classifier checkpoints with their saved predictions, saved YOLO boxes, tooth boxes, data splits and the results table. |
| DENTEX | Public dataset on Hugging Face (`ibrahimhamamci/DENTEX`), licence CC BY-NC-SA 4.0. Downloaded the way the notebook's data cell does (`hf_hub_download` + `dentex_utils.safe_extract`), without the self-training pool. |

DENTEX as parsed by the notebook's own `dentex_utils.discover`: 705 training X-rays with diagnosis labels
(3,529 boxes: caries 2,189, deep caries 578, periapical lesion 158, impacted 604), 50 official validation X-rays
(182 boxes: caries 101, deep caries 32, periapical lesion 9, impacted 40), 634 enumeration X-rays (tooth labels only),
and 250 test X-rays whose released labels use a different Turkish label set; the notebook does not use those, and
neither does this work.

Protocol A keys (`assets/keys`): 635 training, 70 validation (settings), 50 test (= all 50 official validation
X-rays). Re-deriving the split from DENTEX with the notebook's rule (`random.Random(123)` shuffle of the sorted
training keys) gives exactly the same keys. A fourth key file lists the 635 training X-rays plus 1,961 X-rays without
diagnosis labels that the self-training rounds used.

### 2.2 Step 0: inventory before building

Before any code was written, every file was mapped to a component of the system and every setting was read from
`results_protocol_A_v15.csv` (the `chosen` and `boxes` columns of the rows picked on validation). The mapping was
then proven by an offline replay: the final fusion was rebuilt from the **saved** per-model outputs alone, with no
model run, and it reproduced the reported numbers exactly (48.77 AP, 69.40 AP50, validation 48.69), as did each of
the seven YOLOs and both classifiers. The replay also settled one ambiguity: the median box adapter is fitted on the
clean tooth detector's boxes (`teeth_clean.pkl`), which gives 32.21 AP for the crop-only classifier as reported,
whereas fitting it on `teeth.pkl` gives 31.75.

Library versions were agreed with the author from the Kaggle logs: Python 3.13, torch 2.11.0, ultralytics 8.4.174.
Two checkpoints (`disA_r1280.pt`, `disA_st1280_t4tta.pt`) were saved by ultralytics 8.4.175 during an earlier run;
inference stays on 8.4.174 as in every v15 session.

---

## 3. The deployed system (protocol A of v15)

| # | Component | File in the release | Details (as run in the notebook) |
|---|---|---|---|
| 1 | Clean tooth detector | `weights/tooth_clean.pt` | YOLO11s, 32 classes (one per FDI number 11 to 48), input 1024 px, confidence 0.10, class-agnostic NMS IoU 0.5, at most 64 boxes; the most confident box per tooth number is kept. Trained only on enumeration X-rays that are not (near) copies of any diagnosis X-ray (the v15 audit), so it never saw a validation or test X-ray. |
| 2 | Tooth crops | (code) | Each tooth box enlarged by 15% of its width on both sides, 8% of its height on the crown side and 20% on the root side (up in the upper jaw, down in the lower jaw); letterboxed to 384 x 224 (height x width), grey; crops of quadrants 2 and 3 mirrored so both sides of the jaw look alike. Cut from the original, full-resolution X-ray. |
| 3 | Tooth classifier | `cls/v9_A_single/best.pt` | ConvNeXt-nano (`convnext_nano.in12k_ft_in1k`) with a tooth-number embedding, 4 sigmoid outputs, variant "crop only", seed 0, 15.8 M parameters. Boxes from the **median box adapter**, confidence power **0.0** (both picked on validation). |
| 4 | Self-trained tooth classifier | `cls/v11_A_single_st_t4tta/best.pt` | Same architecture, round-1 student (trained on the 635 X-rays plus pseudo-labelled ones). Median box adapter, confidence power 0.0. |
| 5 | Three plain YOLO11m | `weights/disA.pt`, `disA_s1.pt`, `disA_r1280.pt` | 1024 px seed 0, 1024 px seed 1, 1280 px. Boxes exactly as the ultralytics validator makes them: confidence 0.001, NMS IoU 0.7, up to 300 boxes, multi-label. |
| 6 | Four self-trained YOLOs | `weights/disA_st1280_t4tta.pt`, `disA_st1280_r2t6tta.pt`, `disA_st1280_r2t6tta_y26.pt`, `disA_st1536_r3t8tta_hi.pt` | Round 1 YOLO11m at 1280; round 2 YOLO11m and YOLO26m at 1280; round 3 YOLO26m at 1536 (reads a second export of the X-ray with the long side at 1536 px). |
| 7 | Final fusion | (results row) | Weighted boxes fusion of the nine members in this order: crop only, self-trained crop only, the three plain YOLOs, the four self-trained YOLOs. All weights 1, IoU 0.6, **detector boxes** on: each classifier finding moves onto the box of the detectors' plain fusion (equal weights, IoU 0.55, all seven YOLOs) for the same tooth when that box lies mostly inside the tooth (containment at least 0.5) and overlaps the finding's own box (IoU at least 0.5). No test-time augmentation (validation chose round 3 without it). |

Quick mode runs only the round-1 self-trained YOLO11m at 1280 px (48.08 AP in the report), plus the tooth detector,
which supplies the tooth numbers and the panoramic check but does not change any score or box.

The median box adapter (8 entries: 4 findings x upper or lower jaw; offsets of each box edge from the tooth box, in
tooth widths and heights) is computed with the notebook's `fit_box_adapter` on the 635 training X-rays and shipped as
`thirtytwo/settings/adapter.json`, because the deployed app has no DENTEX. Example: caries, lower jaw:
-0.0041, -0.0186, -0.0018, -0.0726.

Preprocessing is the notebook's export: grey image, JPEG with the long side at 1280 px (quality 95), and a second
JPEG at 1536 px for the round-3 YOLO.

---

## 4. Implementation: the `thirtytwo` package

About 840 lines of Python, plus the notebook's two modules, imported unchanged.

| Module | Role |
|---|---|
| `config.py` | Every setting of section 3, each with the notebook cell or results row it comes from. |
| `images.py` | Decoding uploads (JPEG, PNG, DICOM) and the export, in memory. |
| `detectors.py` | Tooth detector, and the disease YOLOs run the way the validator runs them. |
| `classifier.py` | Tooth crops and the two classifiers. |
| `fusion.py` | The notebook's glue between models, and the finding-to-tooth rule. |
| `pipeline.py` | `ThirtyTwo`: loads all ten models once (CPU, or a GPU if there is one); `predict(image, mode)`. |
| `predict.py` | Command line: `python -m thirtytwo.predict xray.png --mode full` prints JSON. |
| `assets.py` | Downloads `cusp_assets.zip` at first start (GitHub release, or a Hugging Face model repo), checks the SHA-256 and unpacks only the 10 weight files needed. |
| `server.py` | The FastAPI web app. |

### 4.1 Key engineering decisions

1. **Exact preprocessing without touching the disk.** The notebook's `read_gray` and `resize_long` read from and write
   to files; uploads must stay in memory. They were copied step for step with `cv2.imdecode` / `cv2.imencode`
   (same INTER_AREA resize and rounding, JPEG quality 95), and the JPEG bytes are decoded again the way ultralytics
   reads a file (`IMREAD_COLOR`), because the detectors only ever saw those JPEGs. A test checks that the bytes are
   identical to `resize_long`'s file at 1280 and 1536 px.
2. **Copying how each saved reference was made, for a single image.** The saved tooth boxes came from `model.predict`
   on chunks of 16 mixed-size X-rays, so ultralytics padded nearly every image to a full 1024 x 1024 square;
   `predict(rect=False)` does the same for one image. The saved YOLO boxes came from `model.val`, which pads each
   batch to a rectangular shape with half a stride extra (a 1280 model sees a 1312-pixel-wide canvas) and uses
   multi-label NMS. The app re-implements the ultralytics 8.4.174 validator for a batch of one: its image loading
   and resize, its rectangle shape, its letterbox, the same model backend (fused, fp32), its NMS, the scaling back
   to the image, and the rounding of its JSON output (boxes to 3 decimals, scores to 5), then divides by the export
   scale as the notebook did. This copy gives **bit-identical** boxes to ultralytics' own `model.val(batch=1)`.
3. **The classifiers run through the notebook's own code.** Crops are cut with `dentex_utils.crop_tooth` and passed to
   `train_arch.predict` through its in-memory crop cache, so mirroring, normalisation and the forward pass are
   `train_arch`'s code. The probabilities are bit-identical to `train_arch.predict` reading the PNG crops that
   `dentex_utils.build_crops` writes.
4. **The fusion uses the notebook's functions.** `make_preds`, `with_detector_boxes` and `wbf_members` (notebook
   cells, not in the modules) were copied with the loop over validation and test removed; everything they call
   (`preds_from_teeth`, `preds_from_disdet`, `transplant_boxes`, `wbf`) is `dentex_utils`' code, unchanged. Working
   per X-ray is equivalent, because the fusion groups boxes by X-ray and finding and keeps the members' order.
5. **Finding to tooth.** Each fused finding goes to the tooth with the same number if their IoU is at least 0.1,
   else to the tooth it overlaps most if that IoU is at least 0.3, else to no tooth: the rule `build_crops` uses to
   put an annotated disease box on a tooth. A test checks it against `build_crops` on all 182 boxes of the official 50.
6. **Panoramic check.** An image is not analysed if it is not 1.5 to 3.2 times wider than tall (all 1,639 DENTEX
   X-rays measured are 1.60 to 2.33) or if fewer than 10 teeth are found (2 of the 755 diagnosis X-rays). The user
   sees a plain warning instead of results.
7. **No telemetry.** ultralytics sends usage statistics by default when online; the package sets `YOLO_OFFLINE=1`
   before importing it, and switches off Hugging Face telemetry.

### 4.2 Output of `predict(image, mode)`

The image size; every finding with its class, score, box in original pixels and FDI tooth number (or none); the
detected teeth with their numbers and boxes; the time taken; a warning for non-panoramic images; and the display
thresholds of the mode. Findings are sorted by score; the web page filters them with the thresholds.

---

## 5. Verification: the accuracy check

`tools/accuracy_check.py` runs the deployed pipeline exactly as the app does (one X-ray at a time, from the PNG
bytes, CPU, fp32) on the 50 official validation X-rays and the 70 protocol-A validation X-rays, scores it with
pycocotools through the notebook's `dentex_eval` (COCO AP over IoU 0.50 to 0.95, all boxes), and compares every
member with its saved output from Kaggle. The full run took about 25 minutes on a 4-core CPU. Report:
`docs/accuracy-check.md`.

### 5.1 Results

| Official 50 validation X-rays | AP | AP50 | AP75 | Report AP / AP50 | Result |
|---|---|---|---|---|---|
| Full system | **48.77** | **69.49** | 58.62 | 48.77 / 69.40 | pass (target 48.8 / 69.4 +- 0.3) |
| Quick mode | **48.15** | 69.57 | 58.15 | 48.08 / 69.49 | pass (target 48.1 +- 0.3) |
| Saved Kaggle outputs, fused by the app's code | 48.77 | 69.40 | 58.53 | | identical to the report |

On the 70 protocol-A validation X-rays: full 48.55 AP (report 48.69), Quick 44.39 (report 44.40).

AP per finding (full system, official 50, report values): caries 41.6, deep caries 56.1, periapical lesion 37.7,
impacted tooth 59.6. For comparison the plain YOLO11m: 31.5, 49.7, 14.9, 55.7.

| Each member alone, official 50 | AP | Report AP | Difference |
|---|---|---|---|
| crop only classifier | 32.21 | 32.21 | +0.00 |
| self-trained crop only classifier | 34.92 | 34.91 | +0.01 |
| plain YOLO (1024, seed 0) | 37.86 | 37.97 | -0.11 |
| plain YOLO, seed 1 | 40.55 | 40.52 | +0.03 |
| plain YOLO at 1280 | 39.76 | 39.92 | -0.16 |
| round-1 self-trained YOLO11m | 48.15 | 48.08 | +0.07 |
| round-2 self-trained YOLO11m | 46.64 | 46.63 | +0.01 |
| round-2 self-trained YOLO26m | 46.72 | 46.74 | -0.02 |
| round-3 self-trained YOLO26m | 48.13 | 48.03 | +0.10 |

### 5.2 Where the app differs from the saved outputs, and why

| Stage | Difference (official 50) | Cause |
|---|---|---|
| Tooth detector | Same tooth numbers on all 50 X-rays; over 1,328 teeth the largest box difference is 0.0005 px and the lowest IoU 0.99999 | GPU vs CPU floating point |
| Classifiers | Probabilities differ by a median of 0.000002 and at most 0.01 | The notebook ran them with fp16 autocast on GPU; the app runs fp32 on CPU |
| YOLOs | Most boxes identical (median IoU 1.0, median confidence difference 0); the few that move are almost all low-confidence boxes: of the boxes with confidence at least 0.1, at most 1 per model moves to IoU below 0.9 | Batch padding: the validator padded each batch of 16 (8 above 1024 px) to a shape set by the other X-rays in it; the app sees one X-ray at a time |

These residual differences change each member's AP by at most 0.16 and the final AP by less than 0.01.

### 5.3 Display thresholds

The page shows a finding only if its score reaches a threshold for its class and mode: the score with the best F1 at
IoU 0.5 on the 70 protocol-A validation X-rays (never on the official 50), from the notebook's `f1_threshold`.
Stored in `thirtytwo/settings/thresholds.json`. A sensitivity slider scales all four between x0.5 and x2.

| Finding | Full: threshold (F1) | Quick: threshold (F1) |
|---|---|---|
| Caries | 0.357 (0.567) | 0.250 (0.559) |
| Deep caries | 0.211 (0.634) | 0.258 (0.610) |
| Periapical lesion | 0.106 (0.579) | 0.121 (0.514) |
| Impacted tooth | 0.190 (0.863) | 0.345 (0.851) |

### 5.4 Speed

Measured over the 50 official X-rays on a 4-core CPU with the models loaded: **Quick 1.2 s** per X-ray (slowest
1.5 s), **full 10.5 s** (slowest 13.4 s). Loading the ten models takes 4 to 40 seconds at start. Because the full
system takes more than about 10 seconds, the page starts in Quick mode and shows the expected wait for Full. The
expected times start from these measurements and then follow the server's own recent runs (a mode's first, cold run
is left out); the tool and the How it works page show the same numbers.

---

## 6. The web application

### 6.1 API (FastAPI)

- `POST /api/predict`: multipart form with the image (JPEG, PNG or DICOM, at most 20 MB) and the mode (`quick` or
  `full`); returns the JSON of section 4.2. DICOM files are converted with pydicom (modality and VOI LUT, scaled to
  8 bits, MONOCHROME1 inverted) and get a JPEG preview back, since browsers cannot show DICOM.
- `GET /api/health`: model status, device, default mode, queue length and expected time per mode.
- Uploads are processed in memory and never written to disk or to the logs. Starlette's form parser spools files
  above 1 MB to a temporary file; the app uses a subclass whose buffer never rolls over below the size cap, and a
  test fails if any temporary file is about to be written.
- One prediction runs at a time; up to 4 more wait in line, beyond that the server answers 503 (protects memory).
- A Content Security Policy limits the page to this server (no outside scripts, fonts, images or requests); the API
  answers with `Cache-Control: no-store`.
- At start the server runs one Quick prediction on a bundled sample, so no visitor waits for the warm-up.

### 6.2 Frontend

Plain HTML, CSS and JavaScript served by FastAPI (no frameworks, component libraries, CDNs, external fonts or
analytics); about 1,200 lines in total.

- **Tool page (`/`).** Title line and a key of the four findings (colour and code). A drop zone ("Drop a panoramic
  X-ray here, or choose a file"), two sample X-rays and the privacy line. After an upload: a viewer on a near-black
  panel with wheel and pinch zoom, drag to pan, keyboard zoom and pan, a reset button, brightness and contrast
  sliders and a "show all teeth" toggle. Boxes are 1.5 px outlines in the class colour, each labelled with the tooth
  number and a code (C, DC, PL, IT), so colour is never the only cue. Next to it, a findings panel: a summary
  ("10 findings on 9 teeth"), a list grouped by finding (swatch, finding, tooth, confidence in %), linked both ways
  with the boxes (hover or focus a row to highlight its box, click a box to select its row), a show or hide toggle
  per finding, the sensitivity slider (x0.5 to x2), a Quick or Full switch with the measured time, and downloads
  (PNG with the boxes, JSON of the shown findings). A progress line with the expected wait while a prediction runs;
  switching mode keeps the results on screen and shows the progress next to the switch. A plain warning replaces
  the results when the image does not look like a panoramic X-ray. The disclaimer "Research prototype. Not a medical
  device. Don't use it to diagnose or treat anyone." sits under the viewer.
- **How it works (`/how-it-works`).** A four-step diagram and cards: a tooth detector finds and numbers every tooth;
  a classifier checks each tooth up close; detectors scan the whole X-ray; nine models vote, and boxes that several
  models agree on score higher. A comparison of Quick and Full (time and AP).
- **Accuracy (`/accuracy`).** The headline numbers (48.8 full, 48.1 Quick, 38.0 standard YOLO11m, 47.8 average over
  five random splits), a one-sentence explanation that AP is a strict 0 to 100 box-overlap score and not percent
  correct, two bar figures (the three systems; AP per finding), limitations and how the numbers were measured.
- **About (`/about`).** The name, what each of the four findings is, a chart of the FDI tooth numbers drawn as a
  panoramic X-ray shows the mouth, what the project is for, privacy, data and licence, and the DENTEX citation.
- The top bar shows the name; the footer links the four pages and marks the current one.

### 6.3 Design rules followed

Background #FAFAF9, text #111, 1 px borders #E5E5E2, corner radius at most 6 px, no shadows, gradients, blur or glow,
one accent colour (#1B5FAA, 6.2:1 contrast on the background) for links and the main button only, system fonts with
one type scale, tabular numbers for percentages, sentence case, no emojis, three line icons (upload, download,
reset). Usable with the keyboard with visible focus; works at 375 px wide without sideways scrolling.

Class colours are the colour-blind-safe Okabe-Ito set: caries #E69F00, deep caries #CC79A7, periapical lesion
#56B4E9, impacted tooth #F0E442. `tools/check_colours.py` measured the closest pair of colours (CIEDE2000) under
normal vision (21.7) and simulated protanopia (14.0), deuteranopia (11.6) and tritanopia (10.9, caries vs deep
caries); about 10 is the usual threshold for colours that are clearly different side by side. Every colour has at
least 6.3:1 contrast on the dark viewer, and the dark label text has at least 6.2:1 on every colour. The
per-finding bar figure therefore sits on the same near-black panel as the viewer; on a light page the colours would
fall below 3:1.

---

## 7. Testing

- **pytest, 23 tests, all passing.** Parity with the notebook's code: the JPEG export byte for byte against
  `resize_long` at 1280 and 1536 px; decoding against `read_gray`; the crops against `build_crops`; the classifier
  against `train_arch.predict` on saved PNG crops (bit-identical); two YOLOs against ultralytics' own validator
  (identical); the tooth detector against `predict` on the JPEG file (identical); the fusion code on the saved
  outputs (48.77 / 69.40 / validation 48.69); the adapter JSON against its rebuild (same table, 32.21 AP); the
  finding-to-tooth rule against `build_crops` on the official 50; Quick mode equal to the round-1 member of the full
  run. API: health and pages, both modes, JPEG, PNG and DICOM uploads, the 20 MB cap (413), wrong file types (415),
  bad mode or broken image (422), the in-memory guard, the panoramic warning, and one prediction at a time with a
  bounded queue. Tests that need DENTEX or the saved Kaggle outputs are skipped when those are missing.
- **Browser check** (`tools/screenshots.js`, Playwright): all four pages at 375 x 812 and 1440 x 900; a sample
  analysed in Quick mode and then in Full mode; zoom and "show all teeth"; keyboard focus on a findings row. It fails
  on console errors, requests to other hosts, failed requests, sideways scrolling or elements sticking out of the
  page. Result: no problems.
- **First-start download** tested into an empty folder: 23 seconds to download, verify and unpack the weights, which
  were byte-identical to the tested ones.

---

## 8. Packaging and deployment

- `requirements.txt` pins torch 2.11.0, torchvision 0.26.0, ultralytics 8.4.174, ultralytics-thop 2.2.3,
  timm 1.0.30, opencv-python-headless 5.0.0.93, numpy 2.5.3, pandas 3.0.5, pillow 12.3.0, pydicom 3.0.2,
  huggingface_hub 2.4.0, safetensors 0.8.0, fastapi 0.162.0, starlette 1.7.0, uvicorn 0.54.0,
  python-multipart 0.0.32 (Python 3.12 or 3.13; numpy 2.5.3 needs at least 3.12). The CPU wheels of torch come from
  the PyTorch index.
- `make run` creates a virtual environment, installs everything and starts the app on port 7860; `make test` and
  `make check` run the tests and the accuracy check.
- The `Dockerfile` (Python 3.13, CPU torch, user 1000, port 7860) is ready for Hugging Face Spaces (Docker SDK, free
  CPU); the README gives the steps. Weights are never in git: the app downloads `cusp_assets.zip` from the GitHub
  release at first start, or from a Hugging Face model repo if `THIRTYTWO_ASSETS_HF_REPO` is set.
- Not verified in the development sandbox: building the Docker image (Docker Hub refused anonymous pulls with 429,
  and the PyTorch download site was blocked by the sandbox's network policy).

---

## 9. Privacy, ethics and licence

- X-rays are processed in memory and never stored or logged; the page loads nothing from other websites; telemetry
  of the libraries is off.
- The app is a research prototype, not a medical device, and says so next to the results.
- DENTEX is released under CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike), so Thirty Two is for research
  and teaching only, as the About page states. The repository contains no DENTEX images except two unmodified
  validation X-rays (`val_14.png`, `val_28.png`, chosen because each has all four kinds of finding) used as samples,
  with the DENTEX credit and the licence notice next to them (`thirtytwo/static/samples/LICENSE.txt`). The About page
  cites DENTEX (Hamamci et al., arXiv:2305.19112, 2023).

---

## 10. Limitations

- Trained and tested only on DENTEX (three hospitals in one country); not tested on other X-ray machines or clinics.
- It can miss findings and raise false alarms; periapical lesions are the hardest class (37.7 AP).
- On a CPU, results differ from the GPU runs by floating-point noise and by batch padding (section 5.2); the measured
  effect on the final AP is below 0.01.
- The display thresholds were chosen on 70 X-rays and are therefore approximate.
- DICOM support is a convenience: the models only ever saw DENTEX PNG X-rays.
- Speed depends on the machine: about 10 s per X-ray for the full system on a 4-core CPU, roughly twice that on the
  free 2-core Hugging Face hardware.

## 11. Possible future work

- Faster CPU inference (for example exporting the detectors to ONNX or OpenVINO), so that Full could become the default.
- Testing on X-rays from other machines and clinics.
- Calibrated scores, so that the confidence shown is a probability.
- An optional test-time augmentation mode, which the notebook evaluated.

---

## 12. Files and history

| Path | Purpose |
|---|---|
| `thirtytwo/` | The package and the web app (section 4); `settings/` holds `adapter.json` and `thresholds.json`; `static/` the pages, styles, scripts and the two samples |
| `tools/accuracy_check.py` | The accuracy check (section 5); writes `docs/accuracy-check.md` and `thresholds.json` |
| `tools/references.py` | Loads the saved Kaggle outputs and replays the fusion from them |
| `tools/build_adapter.py` | Rebuilds `adapter.json` |
| `tools/get_dentex.py` | Downloads DENTEX the way the notebook does |
| `tools/check_colours.py` | The colour-blindness and contrast check of the class colours |
| `tools/screenshots.js` | The browser check |
| `tests/` | pytest suite (section 7) |
| `docs/accuracy-check.md` | The full accuracy report, including per-member differences |
| `requirements.txt`, `Dockerfile`, `Makefile`, `README.md` | Packaging and instructions |

Commits on branch `claude/lucid-davinci-mk3xmn` (31 October 2026):

| Commit | Change |
|---|---|
| `4c36af2` | Ignore assets, data, weights and images |
| `71b0097` | The inference package and command line |
| `40eb386` | The API, accuracy check, tests, packaging and the two samples |
| `02ea4c1` | The frontend and the display thresholds |
| `a057580` | Page text and README; browser check in both modes |
| `12b451e` | Separate How it works, Accuracy and About pages |
| `1a4a6b2` | Simpler top bar and footer; one source for the expected times |

## Key numbers at a glance

| Quantity | Value |
|---|---|
| Full system, official 50 (app on CPU) | 48.77 AP, 69.49 AP50 (report 48.77 / 69.40) |
| Quick mode, official 50 (app on CPU) | 48.15 AP (report 48.08) |
| Standard YOLO11m baseline (report) | 38.0 AP |
| Average over five random splits (report) | 47.8 AP |
| Models in the full system | 10 (1 tooth detector, 2 classifiers, 7 YOLOs) |
| Time per X-ray, 4-core CPU | Quick 1.2 s, full 10.5 s |
| Upload limit | 20 MB; JPEG, PNG, DICOM |
| Tests | 23 pytest tests passing; browser check clean at 375 and 1440 px |
| Weights download | 433 MB, SHA-256 checked, about 23 s |
