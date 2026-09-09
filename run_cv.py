"""Live OpenCV demo window. Keys: click floor then kerb-top to calibrate · r = capture empty-box reference · m = toggle tint/reference
· [ ] = rainfall down/up · d = drainage normal/blocked/good · s = save snapshot · q = quit.   python run_cv.py --source 0"""
import argparse, time, cv2, numpy as np
from rig.core import Calibration, WaterDetector, predict_level, fuse, draw
ap = argparse.ArgumentParser(); ap.add_argument("--source", default="0"); ap.add_argument("--config", default="config.json"); ap.add_argument("--rain", type=int, default=30); a = ap.parse_args()
cal = Calibration(a.config); det = WaterDetector(cal); src = int(a.source) if a.source.isdigit() else a.source; cap = cv2.VideoCapture(src)
rain, drains, di = a.rain, ["normal", "blocked", "good"], 0; clicks = []; alerts = []; last_level = None
def on_mouse(ev, x, y, flags, param):
    global clicks
    if ev == cv2.EVENT_LBUTTONDOWN:
        clicks.append(y)
        if len(clicks) == 2:
            h = param["h"]; cal.cfg["floor_y"], cal.cfg["kerb_y"] = max(clicks) / h, min(clicks) / h; cal.save(); clicks = []; print("calibrated: floor", cal.cfg["floor_y"], "kerb", cal.cfg["kerb_y"])
cv2.namedWindow("JalDrishti Rig"); param = {"h": 720}; cv2.setMouseCallback("JalDrishti Rig", on_mouse, param)
while True:
    ok, frame = cap.read()
    if not ok:
        if not str(src).isdigit(): cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        break
    if frame.shape[1] > 1280: frame = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
    param["h"] = frame.shape[0]
    m = det.measure(frame); pred = predict_level(rain, drains[di]); corrected, rule = fuse(pred, m["level"], m["confidence"])
    if corrected in ("KNEE", "WHEEL-DEEP") and corrected != last_level: alerts.append((time.strftime("%H:%M:%S"), corrected, m["real_cm"])); print("ALERT", alerts[-1])
    last_level = corrected
    vis = draw(frame, m, pred, corrected, rule)
    cv2.putText(vis, f"rain {rain} mm/h · drainage {drains[di]} · mode {cal.cfg['mode']} · click floor+kerb to calibrate · r ref · q quit", (10, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    for i, (t, lvl, cm) in enumerate(alerts[-3:]): cv2.putText(vis, f"{t}  {lvl}  {cm:.0f} cm", (vis.shape[1] - 260, 100 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 255), 2)
    cv2.imshow("JalDrishti Rig", vis); k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == ord("r"): det.set_reference(frame); cal.cfg["mode"] = "ref"; cal.save(); print("reference captured")
    if k == ord("m"): cal.cfg["mode"] = {"edge": "tint", "tint": "ref", "ref": "edge"}[cal.cfg["mode"]]; cal.save(); print("mode", cal.cfg["mode"])
    if k == ord("["): rain = max(0, rain - 5)
    if k == ord("]"): rain = min(150, rain + 5)
    if k == ord("d"): di = (di + 1) % 3
    if k == ord("s"): cv2.imwrite(f"snapshot_{int(time.time())}.jpg", vis)
cap.release(); cv2.destroyAllWindows()
