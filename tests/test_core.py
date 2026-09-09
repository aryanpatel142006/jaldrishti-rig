import json, os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rig.core import Calibration, WaterDetector, classify, predict_level, fuse
from tools.make_test_video import frame, FLOOR, KERB_TOP, H, W

def make_det(tmp_path):
    cal = Calibration(str(tmp_path / "cfg.json")); cal.cfg.update({"floor_y": FLOOR / H, "kerb_y": KERB_TOP / H, "x0": 0.1, "x1": 0.9, "top_y": 0.2, "smooth": 1, "mode": "tint"}); return WaterDetector(cal)

def test_classes():
    assert classify(0)[0] == "DRY" and classify(10)[0] == "ANKLE" and classify(30)[0] == "KNEE" and classify(80)[0] == "WHEEL-DEEP"

def test_depth_on_synthetic_frames(tmp_path):
    det = make_det(tmp_path); rng = np.random.default_rng(1); errs = []
    for t in (3, 10, 20, 30, 38):
        img, truth = frame(t, rng); m = det.measure(img); errs.append(abs(m["real_cm"] - truth))
        assert m["level"] == classify(truth)[0], (t, truth, m)
    assert np.mean(errs) < 2.0

def test_edge_mode_on_tinted_tray(tmp_path):
    det = make_det(tmp_path); det.cal.cfg["mode"] = "edge"; rng = np.random.default_rng(3)
    assert det.measure(frame(3, rng)[0])["level"] == "DRY"                      # kerb/road edges must not be mistaken for water
    for t in (12, 22, 32):
        img, truth = frame(t, rng); m = det.measure(img); assert abs(m["real_cm"] - truth) < 3.0 and m["level"] == classify(truth)[0], (t, truth, m)

def test_reference_mode(tmp_path):
    det = make_det(tmp_path); det.cal.cfg["mode"] = "ref"; rng = np.random.default_rng(2)
    empty, _ = frame(2, rng); det.set_reference(empty); wet, truth = frame(25, rng); m = det.measure(wet)
    assert abs(m["real_cm"] - truth) < 3.0 and m["level"] == classify(truth)[0]

def synth_tank(depth_cm, floor=780, ppc=26.0, W=1600, H=900, seed=0):
    """Side-view glass tank with murky (untinted) water, printed marks, dark car."""
    rng = np.random.default_rng(seed); img = np.full((H, W, 3), (215, 210, 205), np.uint8); cv2.rectangle(img, (60, 60), (W - 60, floor), (200, 196, 190), -1)
    img[60:floor, 60:W - 60] = np.clip(img[60:floor, 60:W - 60].astype(int) + rng.normal(0, 6, (floor - 60, W - 120, 3)), 0, 255).astype(np.uint8)
    for cm in range(0, 25): cv2.line(img, (70, int(floor - cm * ppc)), (95 if cm % 5 else 120, int(floor - cm * ppc)), (40, 40, 40), 2)
    cv2.rectangle(img, (60, floor), (W - 60, floor + 40), (70, 70, 75), -1); cv2.rectangle(img, (500, floor - 180), (1124, floor - 20), (120, 70, 40), -1)
    ys = int(floor - depth_cm * ppc); murky = np.full_like(img, (90, 130, 165)); m = np.zeros((H, W), bool); m[ys:floor, 60:W - 60] = True
    return np.where(m[..., None], (0.55 * murky + 0.45 * img).astype(np.uint8), img)

def test_edge_mode_on_murky_side_view(tmp_path):
    cal = Calibration(str(tmp_path / "c.json")); cal.cfg.update({"floor_y": 780 / 900, "kerb_y": (780 - 10 * 26) / 900, "kerb_cm": 10.0, "x0": 0.1, "x1": 0.95, "top_y": 0.08, "mode": "edge", "smooth": 1})
    det = WaterDetector(cal); errs = []
    for d in (3.0, 6.5, 10.0, 14.0):
        m = det.measure(synth_tank(d)); errs.append(abs(m["real_cm"] - d))
    assert max(errs) < 0.6, errs
    assert det.measure(synth_tank(0.0))["real_cm"] < 1.0                     # empty tank reads dry

def test_prediction_and_fusion():
    assert predict_level(10) == "DRY" and predict_level(30) == "ANKLE" and predict_level(50) == "KNEE" and predict_level(90) == "WHEEL-DEEP"
    assert predict_level(30, "blocked") == "KNEE"
    assert fuse("ANKLE", "KNEE", 0.9)[0] == "KNEE"            # observation overrides upward
    assert fuse("WHEEL-DEEP", "DRY", 0.9)[0] == "KNEE"        # far below prediction -> corrected one step down
    assert fuse("KNEE", "ANKLE", 0.3)[0] == "KNEE"            # unreliable observation -> prediction kept
