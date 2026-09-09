"""Streamlit dashboard for the rig (blueprint section 11). Camera is read server-side (laptop webcam or a video file)."""
import time, cv2, numpy as np, pandas as pd, streamlit as st
from rig.core import Calibration, WaterDetector, predict_level, fuse, draw, CLASSES, ORDER
st.set_page_config(page_title="JalDrishti Rig — flood depth from the demo model", page_icon="🌊", layout="wide")
st.markdown("<style>.big{font-size:40px;font-weight:800;margin:0}.lbl{color:#5b6b7c;font-size:12px;text-transform:uppercase;letter-spacing:1px}.card{background:#fff;border:1px solid #dfe6ee;border-radius:12px;padding:12px 14px}.flag{background:#fdecea;border:1px solid #f5c2c0;border-radius:8px;padding:8px 10px;font-weight:600}</style>", unsafe_allow_html=True)
st.title("🌊 JalDrishti Monitoring Dashboard — physical rig")
cal = Calibration("config.json"); ss = st.session_state; ss.setdefault("alerts", []); ss.setdefault("run", False); ss.setdefault("ref", None); ss.setdefault("last", None)
with st.sidebar:
    st.header("Camera"); source = st.text_input("Source (0 = webcam, or a video path)", "0"); run = st.toggle("Live", value=ss.run); ss.run = run
    st.header("Calibration (drag lines onto the rig)")
    cal.cfg["floor_y"] = st.slider("Road / floor line (% from top)", 0.3, 1.0, float(cal.cfg["floor_y"]), 0.005); cal.cfg["kerb_y"] = st.slider("Mark line — kerb top / printed cm line (% from top)", 0.05, 0.95, float(cal.cfg["kerb_y"]), 0.005)
    cal.cfg["x0"], cal.cfg["x1"] = st.slider("Measure across x (%)", 0.0, 1.0, (float(cal.cfg["x0"]), float(cal.cfg["x1"])), 0.01); cal.cfg["top_y"] = st.slider("Search up to (% from top)", 0.0, 0.6, float(cal.cfg["top_y"]), 0.01)
    cal.cfg["kerb_cm"] = st.number_input("Mark height (cm): kerb top = 15, or a printed line", 1.0, 100.0, float(cal.cfg["kerb_cm"]), 1.0); cal.cfg["tyre_cm"] = st.number_input("Tyre represents (cm)", 20.0, 120.0, float(cal.cfg["tyre_cm"]), 5.0)
    modes = ["Water surface edge (side view, any water)", "Blue-tinted water", "Compare with empty-box reference"]; mode = st.radio("Water detection", modes, index={"edge": 0, "tint": 1, "ref": 2}[cal.cfg["mode"]]); cal.cfg["mode"] = ["edge", "tint", "ref"][modes.index(mode)]
    cal.cfg["min_frac"] = st.slider("Sensitivity", 0.2, 0.8, float(cal.cfg["min_frac"]), 0.05)
    capture_ref = st.button("📷 Capture empty-box reference"); st.button("💾 Save calibration", on_click=cal.save)
    st.header("Prediction module"); rain = st.slider("Forecast rainfall (mm/h)", 0, 120, 30, 5); drainage = st.selectbox("Drainage state", ["normal", "blocked", "good"])
    if st.button("Clear alert log"): ss.alerts = []
det = WaterDetector(cal)
if ss.ref is not None: det.ref = ss.ref
left, right = st.columns([3, 2]); frame_slot = left.empty(); cards = right.empty(); log_slot = st.empty()
def render(m, frame):
    pred = predict_level(rain, drainage); corrected, rule = fuse(pred, m["level"], m["confidence"])
    frame_slot.image(cv2.cvtColor(draw(frame, m, pred, corrected, rule), cv2.COLOR_BGR2RGB), width='stretch')
    col = {"DRY": "#2E8B57", "ANKLE": "#F2C230", "KNEE": "#E07B1F", "WHEEL-DEEP": "#C0392B"}
    action = CLASSES[ORDER[corrected]][3]
    html = (f'<div class="card"><div class="lbl">Predicted level (rainfall {rain} mm/h, drainage {drainage})</div><p class="big" style="color:{col[pred]}">{pred}</p></div><br>'
            f'<div class="card"><div class="lbl">Observed CCTV level</div><p class="big" style="color:{col[m["level"]]}">{m["level"]}</p><div class="lbl" style="text-transform:none">{m["real_cm"]:.0f} cm real · kerb {m["kerb_submerged"]*100:.0f}% · tyre {m["tyre_submerged"]*100:.0f}% · conf {m["confidence"]:.2f}</div></div><br>'
            f'<div class="card"><div class="lbl">Corrected nowcast</div><p class="big" style="color:{col[corrected]}">{corrected}</p><div class="lbl" style="text-transform:none">{rule}</div></div><br>'
            + (f'<div class="flag">🚨 RECOMMENDED ACTION: {action}</div>' if corrected in ("KNEE", "WHEEL-DEEP") else f'<div class="card">✅ {action}</div>'))
    cards.markdown(html, unsafe_allow_html=True)
    if corrected in ("KNEE", "WHEEL-DEEP") and (not ss.alerts or ss.alerts[-1]["level"] != corrected):
        ss.alerts.append({"time": time.strftime("%H:%M:%S"), "level": corrected, "observed": m["level"], "predicted": pred, "depth_cm": m["real_cm"], "action": action})
    if ss.alerts: log_slot.dataframe(pd.DataFrame(list(reversed(ss.alerts))[:10]), width='stretch', hide_index=True)
src = int(source) if source.isdigit() else source
if run:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): st.error(f"Cannot open camera/video: {source}")
    else:
        n = 0
        while ss.run and n < 100000:
            ok, frame = cap.read()
            if not ok:
                if not str(src).isdigit(): cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                st.error("Camera read failed"); break
            if frame.shape[1] > 1280: frame = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1])))
            if capture_ref and n == 0: det.set_reference(frame); ss.ref = frame.copy(); cal.cfg["mode"] = "ref"; cal.save()
            m = det.measure(frame); ss.last = (m, frame); render(m, frame); n += 1; time.sleep(0.03)
        cap.release()
else:
    up = st.file_uploader("…or analyse a single photo of the rig", type=["jpg", "jpeg", "png"])
    if up:
        frame = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR); frame = cv2.resize(frame, (1280, int(frame.shape[0] * 1280 / frame.shape[1]))) if frame.shape[1] > 1280 else frame
        if capture_ref: det.set_reference(frame); ss.ref = frame.copy(); cal.cfg["mode"] = "ref"; cal.save(); st.success("Reference stored")
        det.hist.clear(); render(det.measure(frame), frame)
    elif ss.last: render(*ss.last)
    else: st.info("Toggle **Live** in the sidebar (webcam), enter a video path, or upload a photo of the rig.")
