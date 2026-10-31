---
title: Thirty Two
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
short_description: Caries, deep caries, periapical lesions, impacted teeth
---

# Thirty Two

Thirty Two finds four problems on a panoramic dental X-ray: caries, deep caries, periapical lesions and impacted
teeth. You upload an X-ray and see each finding as a box on the image, with its tooth number. It is named after the
32 adult teeth.

It runs the system of the v15 notebook in this repository (`dentex_twinarch_kaggle_2xT4.ipynb`, protocol A: trained
on 635 DENTEX X-rays, every setting picked on 70 held-out ones), with the notebook's own code
(`dentex_utils.py`, `train_arch.py`) and its trained weights.

**Research prototype. Not a medical device. Don't use it to diagnose or treat anyone.** DENTEX is licensed for
non-commercial use (CC BY-NC-SA 4.0), so Thirty Two is for research and teaching only.

## Accuracy

On the 50 official DENTEX validation X-rays (COCO AP over IoU 0.50 to 0.95, pycocotools):

| | AP | AP50 |
|---|---|---|
| Full system | 48.8 | 69.4 |
| Quick mode | 48.1 | 69.5 |
| Standard YOLO11m | 38.0 | 55.7 |

This app reproduces these numbers on a CPU; see [docs/accuracy-check.md](docs/accuracy-check.md) for the check, the
comparison of every model with its saved Kaggle outputs, and how the display thresholds were set.

## How it works

1. A tooth detector (YOLO11s, one class per FDI tooth number) finds and numbers every tooth.
2. A classifier (ConvNeXt-nano) checks each tooth up close, on a crop with extra room at the root. Two of them run:
   the original and a self-trained one.
3. Seven YOLO detectors scan the whole X-ray: three plain ones and four self-trained ones.
4. The nine models vote by weighted boxes fusion: boxes that several models agree on score higher. The classifiers'
   findings sit on the detectors' box for the same tooth.

Some of the models were trained further on 1,961 extra DENTEX X-rays that the system labelled itself. Quick mode runs
only the round-1 self-trained YOLO11m (plus the tooth detector, for the tooth numbers).

## Run it locally

Python 3.13 and `make`:

```bash
make run
```

The first run creates `.venv`, installs the pinned packages (CPU wheels of torch 2.11.0) and starts the app on
http://127.0.0.1:7860. At its first start the app downloads its weights, `cusp_assets.zip` (433 MB), from this
repository's GitHub release `assets-v15`, checks the SHA-256 and unpacks the 10 files it needs into `assets/`.

With Docker instead:

```bash
make docker
```

### Command line and Python

```bash
python -m thirtytwo.predict xray.png --mode quick
```

```python
from thirtytwo.pipeline import ThirtyTwo
model = ThirtyTwo()                      # loads every model once; uses a GPU if there is one
result = model.predict('xray.png', 'full')
```

`predict` returns the image size, every finding (class, score, box in original pixels, FDI tooth number or none),
the detected teeth with their numbers and boxes, and the time taken. A finding goes to the tooth it overlaps: the
tooth with the same number if their IoU is at least 0.1, else the tooth with the largest IoU if that is at least 0.3,
else none (the notebook's rule for putting a disease box on a tooth).

### API

* `POST /api/predict`: multipart form with `image` (JPEG, PNG or DICOM, at most 20 MB) and `mode` (`quick` or
  `full`). Returns the JSON above.
* `GET /api/health`: whether the models are loaded, and the expected time per mode.

Uploads are read in memory and never written to disk or to the logs. One prediction runs at a time; a few more wait
in line, beyond that the server answers 503.

## Where the weights come from

The weights are not in git. By default the app downloads
`https://github.com/Sameer6556/ThirtyTwo/releases/download/assets-v15/cusp_assets.zip`. Settings (environment
variables):

| variable | default | |
|---|---|---|
| `THIRTYTWO_ASSETS_URL` | the release URL above | another place to download the zip from |
| `THIRTYTWO_ASSETS_HF_REPO` | empty | a Hugging Face model repo holding `cusp_assets.zip`, used instead of the URL |
| `THIRTYTWO_ASSETS_SHA256` | the hash of the v15 zip | set it to the new hash if you replace the zip, or empty to skip the check |
| `THIRTYTWO_ASSETS` | `assets/` | where the weights are unpacked |
| `THIRTYTWO_DEFAULT_MODE` | `quick` | the mode the page starts in |

To move the weights to Hugging Face later: create a model repo (for example `your-name/thirtytwo-weights`), upload
`cusp_assets.zip` to it, and set `THIRTYTWO_ASSETS_HF_REPO=your-name/thirtytwo-weights` (for a Space: under
Settings, Variables).

## Deploy to Hugging Face Spaces (free CPU)

1. Create a Space with the Docker SDK and the free CPU hardware.
2. Push this repository to it. The block at the top of this README is the Space configuration (Docker, port 7860).

   ```bash
   git remote add space https://huggingface.co/spaces/your-name/thirty-two
   ```

   ```bash
   git push space main
   ```

3. The Space builds the `Dockerfile`, starts, downloads the weights from the GitHub release (about a minute) and
   then answers. The free Space sleeps when nobody uses it and downloads the weights again after a restart.

The two sample X-rays in `thirtytwo/static/samples` are pushed with the code. Git hosting on Hugging Face may ask to
store them with Git LFS (`git lfs track "*.png"`).

## Check it yourself

The accuracy check needs DENTEX (about 11 GB to download, 6 GB kept), downloaded the way the notebook does:

```bash
python tools/get_dentex.py
```

```bash
make check
```

It writes `docs/accuracy-check.md` and `thirtytwo/settings/thresholds.json`. Tests:

```bash
make test
```

The parity tests compare the app with the notebook's own functions: the in-memory JPEG export against
`dentex_utils.resize_long` (byte for byte, at 1280 and 1536), the crops against `dentex_utils.build_crops`, the
classifier against `train_arch.predict` on saved crops, the YOLOs against ultralytics' own validator, the fusion
against the saved Kaggle outputs (48.77 AP), and the median box adapter against its rebuild (32.21 AP).

## Data and credit

Trained and evaluated on DENTEX (Hamamci et al., 2023), https://huggingface.co/datasets/ibrahimhamamci/DENTEX,
released under CC BY-NC-SA 4.0. The two sample X-rays are unmodified files of the DENTEX validation set and stay
under that licence (see `thirtytwo/static/samples/LICENSE.txt`).

```
@article{hamamci2023dentex,
  title={DENTEX: An Abnormal Tooth Detection with Dental Enumeration and Diagnosis Benchmark for Panoramic X-rays},
  author={Hamamci, Ibrahim Ethem and Er, Sezgin and Simsar, Enis and Yuksel, Atif Emre and Gultekin, Sadullah and
          Ozdemir, Serife Damla and Yang, Kaiyuan and Li, Hongwei Bran and Pati, Sarthak and Stadlinger, Bernd and others},
  journal={arXiv preprint arXiv:2305.19112},
  year={2023}
}
```

Built by Sameer Singh (B.Tech CSE, IIIT Manipur) as his major project, supervised by Dr. Khoirom Motilal Singh.
