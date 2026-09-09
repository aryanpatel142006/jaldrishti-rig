"""Synthetic rig video: translucent tray, black road, striped kerb, red toy car, blue-tinted water rising 0 -> 70 cm real, then draining.
Writes samples/rig_test.mp4 and samples/rig_truth.json (real cm per second)."""
import json, os, numpy as np, cv2
W, H, FPS, SEC = 1280, 720, 15, 60
FLOOR, KERB_TOP = 520, 470                       # kerb = 50 px = 15 cm real -> 3.33 px/cm
def real_cm(t): return 0 if t < 6 else min(70, (t - 6) * 2.5) if t < 40 else max(0, 70 - (t - 40) * 4)
def frame(t, rng):
    img = np.full((H, W, 3), (205, 200, 195), np.uint8); cv2.rectangle(img, (80, 150), (W - 80, FLOOR + 60), (196, 192, 188), -1)   # tray
    cv2.rectangle(img, (100, FLOOR - 40), (W - 100, FLOOR), (35, 35, 35), -1)                                                          # road
    for x in range(140, W - 140, 120): cv2.rectangle(img, (x, FLOOR - 22), (x + 50, FLOOR - 16), (230, 230, 230), -1)
    for i, x in enumerate(range(100, W - 100, 60)): cv2.rectangle(img, (x, KERB_TOP), (x + 60, FLOOR - 40), (20, 20, 20) if i % 2 else (240, 240, 240), -1)   # striped kerb
    cv2.rectangle(img, (560, FLOOR - 62), (720, FLOOR - 40), (40, 40, 200), -1); cv2.rectangle(img, (590, FLOOR - 80), (690, FLOOR - 62), (60, 60, 220), -1)   # toy car
    for cx in (585, 695): cv2.circle(img, (cx, FLOOR - 40), 8, (15, 15, 15), -1)
    img = np.clip(img.astype(np.int16) + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    d = real_cm(t); yw = int(FLOOR - d * (FLOOR - KERB_TOP) / 15.0)
    if d > 0.3:
        ys = np.arange(H)[:, None]; xs = np.arange(W)[None, :]; ripple = 2 * np.sin(xs / 41.0 + t * 2.0)
        mask = (ys >= yw + ripple) & (ys < FLOOR + 60) & (xs > 85) & (xs < W - 85); blue = np.full_like(img, (205, 140, 60))
        img = np.where(mask[..., None], (0.55 * blue + 0.45 * img).astype(np.uint8), img)
    return img, d
if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True); out = cv2.VideoWriter("samples/rig_test.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)); rng = np.random.default_rng(0); truth = []
    for f in range(SEC * FPS):
        img, d = frame(f / FPS, rng); out.write(img)
        if f % FPS == 0: truth.append({"t_s": f // FPS, "real_cm": round(d, 1)})
    out.release(); json.dump(truth, open("samples/rig_truth.json", "w")); print("wrote samples/rig_test.mp4", SEC * FPS, "frames")
