"""JalDrishti Rig — classical-CV flood depth detection for the physical tray model.
Pipeline (blueprint steps): frame -> ROI strip -> water detection (blue tint or background subtraction) -> waterline pixel
-> calibration (kerb top = 15 cm real, floor = 0) -> depth class -> fusion with predicted level -> action + alert.
No deep learning; OpenCV + NumPy only."""
import json, os, time
from collections import deque
import numpy as np, cv2

CLASSES = [  # name, upper bound (cm, real-world), what you see, action
    ("DRY",        2,    "Kerb fully visible, road is dry",                    "No action — monitor only"),
    ("ANKLE",      15,   "Water covers part of the kerb (< 15 cm)",            "Pre-alert · monitor closely · prepare pumps"),
    ("KNEE",       60,   "Kerb submerged, tyre partly under (15–60 cm)",       "Alert · deploy drainage pump · advise caution"),
    ("WHEEL-DEEP", 1e9,  "Tyre fully submerged (> 60 cm)",                     "Close road · traffic diversion · alert police & response"),
]
ORDER = {c[0]: i for i, c in enumerate(CLASSES)}
COLORS = {"DRY": (87, 139, 46), "ANKLE": (48, 194, 242), "KNEE": (31, 123, 224), "WHEEL-DEEP": (43, 57, 192)}   # BGR

def classify(real_cm):
    for name, upper, desc, action in CLASSES:
        if real_cm < upper: return name, desc, action
    return CLASSES[-1][0], CLASSES[-1][2], CLASSES[-1][3]

DEFAULT_CFG = {"floor_y": 0.72, "kerb_y": 0.62, "x0": 0.10, "x1": 0.90, "top_y": 0.15, "kerb_cm": 15.0, "tyre_cm": 60.0,
               "mode": "edge", "hsv_lo": [85, 40, 40], "hsv_hi": [135, 255, 255], "min_frac": 0.45, "diff_thr": 60, "smooth": 7}
# modes: "edge" = horizontal water-surface edge (clear or murky water, side view) · "tint" = blue food colouring · "ref" = compare with empty reference frame
# calibration: floor_y (0 cm) and kerb_y = any mark of known height kerb_cm (the 15 cm kerb, or the 10 cm line printed on a tank)

class Calibration:
    """Fractions of frame height/width so the same calibration works at any resolution. Persisted to config.json."""
    def __init__(self, path="config.json"):
        self.path = path; self.cfg = dict(DEFAULT_CFG)
        if os.path.exists(path):
            try: self.cfg.update(json.load(open(path)))
            except Exception: pass
    def save(self): json.dump(self.cfg, open(self.path, "w"), indent=1)
    def px(self, shape):
        h, w = shape[:2]; c = self.cfg
        return {"floor": int(c["floor_y"] * h), "kerb": int(c["kerb_y"] * h), "top": int(c["top_y"] * h), "x0": int(c["x0"] * w), "x1": int(c["x1"] * w)}

class WaterDetector:
    def __init__(self, cal: Calibration):
        self.cal = cal; self.ref = None; self.hist = deque(maxlen=int(cal.cfg.get("smooth", 7)))
    def set_reference(self, frame): self.ref = frame.copy()
    def wet_rows(self, frame, p):
        """Per-row fraction of 'water' columns inside the strip between top and floor."""
        c = self.cal.cfg; band = frame[p["top"]:p["floor"], p["x0"]:p["x1"]]
        if c["mode"] == "ref" and self.ref is not None:
            rb = self.ref[p["top"]:p["floor"], p["x0"]:p["x1"]]; wet = np.abs(band.astype(np.int16) - rb.astype(np.int16)).sum(axis=2) > c["diff_thr"]
        else:
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV); wet = cv2.inRange(hsv, np.array(c["hsv_lo"], np.uint8), np.array(c["hsv_hi"], np.uint8)) > 0
        prof = wet.mean(axis=1).astype(np.float32)
        return cv2.GaussianBlur(prof.reshape(-1, 1), (1, 7), 0).ravel()
    def surface_edge(self, frame, p):
        """Side-view surface = a horizontal edge that spans (nearly) the whole strip AND has water below it: the rows just below
        the edge are more saturated (murky/tinted water) than the rows just above (air, wall). Row statistics use MEDIANS across
        columns so a car or a pole in the strip cannot fake or hide the surface. Returns (y, conf 0..1)."""
        crop = frame[p["top"]:p["floor"], p["x0"]:p["x1"]]; h, w = crop.shape[:2]
        if h < 30: return p["floor"], 0.0
        g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), 0)
        f = crop.astype(np.float32); sat = cv2.GaussianBlur(f.max(axis=2) - f.min(axis=2), (5, 5), 0)   # chroma (max-min), not HSV saturation: dark noisy pixels stay ~0
        mag = np.abs(np.diff(g, axis=0)) + np.abs(np.diff(sat, axis=0))                 # (h-1, w)
        thr = max(6.0, 2.5 * float(np.median(mag)))
        coverage = cv2.GaussianBlur((mag > thr).mean(axis=1).astype(np.float32).reshape(-1, 1), (1, 9), 0).ravel()
        rs = np.median(sat, axis=1)                                                      # per-row median saturation
        k = 15; cs = np.concatenate([[0.0], np.cumsum(rs)])
        def band_mean(a, b):                                                             # mean of rs[a:b] with clipping, vectorised over candidate rows
            a = np.clip(a, 0, h); b = np.clip(b, 0, h); n = np.maximum(1, b - a); return (cs[b] - cs[a]) / n
        ys = np.arange(h - 1); below = band_mean(ys + 3, ys + 3 + k); above = band_mean(ys - 3 - k, ys - 3)
        contrast = np.clip((below - above) / 25.0, 0, 1)
        score = coverage * (0.4 + contrast); score[: int(0.03 * h)] = 0; score[int(0.98 * h):] = 0
        y = int(np.argmax(score)); sc = float(score[y])
        if coverage[y] < 0.5 or contrast[y] < 0.2: return p["floor"], 0.0             # not a full-width edge with water under it
        return p["top"] + y, float(np.clip((sc - 0.45) / 0.9, 0.2, 1.0))
    def waterline(self, frame):
        """Topmost water row. 'edge' mode: surface edge (side view). 'tint'/'ref': walk up from the floor while rows are wet."""
        p = self.cal.px(frame.shape)
        if self.cal.cfg["mode"] == "edge":
            y, c = self.surface_edge(frame, p)
            if c < 0.15: return p["floor"], 1.0, p         # no convincing surface -> dry
            return y, c, p
        prof = self.wet_rows(frame, p); rows = prof[::-1]; y = 0; gap = 0
        for i, fr in enumerate(rows):
            if fr >= self.cal.cfg["min_frac"]: y = i + 1; gap = 0
            else:
                gap += 1
                if gap > 6: break
        solidity = float(rows[:y].mean()) if y else 1.0
        return p["floor"] - y, solidity, p
    def measure(self, frame):
        yw, sol, p = self.waterline(frame); c = self.cal.cfg
        px_per_cm = (p["floor"] - p["kerb"]) / max(1e-6, c["kerb_cm"])        # the kerb is the ruler: its height = kerb_cm real
        real = max(0.0, (p["floor"] - yw) / px_per_cm)
        self.hist.append(real); real_s = float(np.median(self.hist))
        name, desc, action = classify(real_s)
        conf = float(np.clip(0.4 + 0.55 * sol, 0.4, 0.95)) if real_s > 1 else 0.9
        return {"y_water": int(yw), "real_cm": round(real_s, 1), "raw_cm": round(real, 1), "level": name, "description": desc, "action": action, "confidence": round(conf, 2),
                "kerb_submerged": round(min(1.0, real_s / c["kerb_cm"]), 2), "tyre_submerged": round(min(1.0, real_s / c["tyre_cm"]), 2), "px_per_cm": round(px_per_cm, 2), "p": p, "ts": time.time()}

# ---- prediction + fusion (blueprint sections 8–10) ----
def predict_level(rain_mm_h, drainage="normal"):
    """Sample prediction module: forecast rainfall intensity -> expected street water level. Simple runoff table; blocked drains shift one class up."""
    i = 0 if rain_mm_h < 15 else 1 if rain_mm_h < 40 else 2 if rain_mm_h < 70 else 3
    i = min(3, i + 1) if drainage == "blocked" else max(0, i - 1) if drainage == "good" else i
    return CLASSES[i][0]

def fuse(predicted, observed, obs_conf):
    """A confident camera observation overrides the prediction; if the camera sees far less than predicted, correct the nowcast down one step."""
    p, o = ORDER[predicted], ORDER[observed]
    if obs_conf < 0.5: return predicted, "observation unreliable → prediction kept"
    if o >= p: return observed, "observed ≥ predicted → observation overrides"
    if p - o >= 2: return CLASSES[p - 1][0], "observed ≪ predicted → nowcast corrected one step down"
    return observed, "observed slightly below predicted → observation used"

# ---- drawing ----
def draw(frame, m, predicted=None, corrected=None, rule=None):
    vis = frame.copy(); p = m["p"]; x0, x1, yf, yk, yt, yw = p["x0"], p["x1"], p["floor"], p["kerb"], p["top"], m["y_water"]
    cv2.rectangle(vis, (x0, yt), (x1, yf), (255, 255, 255), 1)
    cv2.line(vis, (x0, yf), (x1, yf), (80, 220, 80), 2); cv2.putText(vis, "floor 0", (x0 + 6, yf + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 2)
    cv2.line(vis, (x0, yk), (x1, yk), (80, 220, 80), 2); cv2.putText(vis, "kerb top 15 cm", (x0 + 6, yk - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 2)
    for cm in (15, 30, 45, 60):
        yy = int(yf - cm * m["px_per_cm"])
        if yy > yt: cv2.line(vis, (x1 - 26, yy), (x1, yy), (80, 220, 80), 1); cv2.putText(vis, str(cm), (x1 - 52, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 220, 80), 1)
    col = COLORS[m["level"]]
    if m["real_cm"] > 1: cv2.line(vis, (x0, yw), (x1, yw), (0, 140, 255), 3); cv2.putText(vis, f'{m["real_cm"]:.0f} cm', (x0 + 6, yw - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 78), (27, 58, 107), -1)
    cv2.putText(vis, f'WATER LEVEL: {m["level"]}   {m["real_cm"]:.0f} cm real   conf {m["confidence"]:.2f}', (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    line2 = m["description"] if predicted is None else f'predicted {predicted}  |  observed {m["level"]}  |  corrected {corrected}'
    cv2.putText(vis, line2, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(vis, ("ACTION: " + (CLASSES[ORDER[corrected]][3] if corrected else m["action"])), (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 220, 255) if (corrected or m["level"]) in ("KNEE", "WHEEL-DEEP") else (200, 200, 200), 1)
    return vis
