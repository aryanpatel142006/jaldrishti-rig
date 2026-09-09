# JalDrishti Rig — flood depth from the physical demo model

The camera-and-tray prototype from the JalDrishti blueprint, as a small standalone program: **a webcam watches a tray with a road, a striped kerb and a toy car; Python + OpenCV finds the water surface, reads the depth against the kerb, classifies it (Dry · Ankle · Knee · Wheel-deep), fuses it with a rainfall-based prediction, and raises the alert.** No deep learning, no ruler, no internet.

```
Physical model ➜ Webcam ➜ OpenCV (ROI, water surface) ➜ Depth vs kerb ➜ Class ➜ Fuse with prediction ➜ Alert
```

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) live OpenCV window (best on stage)
python run_cv.py --source 0                 # 0 = webcam; or a video path, e.g. samples/rig_test.mp4

# 2) dashboard (predicted / observed / corrected + alert log)
streamlit run app.py
```

**Calibrate once** (camera must stay fixed afterwards): in the OpenCV window click the **road/floor** and then the **top of the kerb** — the kerb stands for 15 cm real. In the dashboard, drag the two sliders to the same lines. Saved to `config.json`.

Keys in the OpenCV window: click ×2 = calibrate · `r` = capture empty-box reference · `m` = switch water detection (edge → tint → reference) · `[` `]` = rainfall down/up · `d` = drainage normal/blocked/good · `s` = snapshot · `q` = quit.

## Water detection — three ways, pick per venue

| Mode | When | How |
|---|---|---|
| **edge** (default) | side view, plain or muddy water | finds the horizontal water surface: a full-width edge with more saturated water below than air above; medians across columns so the car cannot fake it |
| **tint** | a few drops of blue food colour in the water | rows whose columns fall in the blue range, walking up from the floor |
| **reference** | untinted water, camera looking down | compares each frame with a captured empty-box frame |

## Depth and classes (from the build guide)

Depth (real cm) = pixels above the floor ÷ pixels-per-cm, where pixels-per-cm comes from the kerb (= 15 cm). Classes: **DRY** < 2 cm · **ANKLE** < 15 cm (kerb partly covered) · **KNEE** 15–60 cm (tyre partly under) · **WHEEL-DEEP** > 60 cm (tyre fully under). Each class carries the action text (pre-alert · deploy pump · close road).

## Prediction + fusion (blueprint §8–10)

`predict_level(rain_mm_h, drainage)` gives the expected level from a rainfall slider. `fuse(predicted, observed, confidence)`: a confident camera reading overrides; if the camera sees far less than predicted, the nowcast is corrected one step down; an unreliable reading keeps the prediction. Knee/Wheel results append to the alert log.

## Test

```bash
python tools/make_test_video.py     # synthetic tray video, water rising 0 → 70 cm real, samples/rig_test.mp4 + truth
python -m pytest -q tests           # classes, depth on synthetic frames (all three modes), prediction, fusion
```

## Files
`rig/core.py` detector, calibration, classes, prediction, fusion, drawing · `run_cv.py` live window · `app.py` Streamlit dashboard · `tools/make_test_video.py` · `tests/` · `config.json` (created on first calibration).

Part of **JalDrishti** (SIH 2026, PS SIH26085) · Team JalDrishti, BVM Engineering College · MIT licence.
