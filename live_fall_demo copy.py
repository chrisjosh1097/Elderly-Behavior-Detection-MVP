import time, os, math, csv, pathlib, datetime
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# --------------------------- CLI CONFIG (quick-tune) ---------------------------
CONF_THRES = 0.25
SEQ_LEN = 32
ALERT_THRESH = 0.80
MIN_FRAMES_FOR_SCORE = 8
COOLDOWN_SEC = 8           # don't log again within this window for same stream

LOG_CSV = "fall_events.csv"
SNAP_DIR = "fall_snaps"    # snapshot folder; set to "" to disable snapshot saving

# Optional classifier (if you trained it)
CLS_WEIGHTS = "fall_lstm_best.pt"
USE_CLASSIFIER = os.path.exists(CLS_WEIGHTS)

# --------------------------- Small utils --------------------------------------
def angle(p1, p2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    return math.degrees(math.atan2(dy, dx))

def get_joint(kpts, idx):
    if kpts is None or idx >= kpts.shape[0] or kpts[idx,2] < 0.2:
        return None
    return (float(kpts[idx,0]), float(kpts[idx,1]))

def largest_person_result(res):
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return None
    # Torch-only argmax (GPU-safe)
    areas = (boxes.xyxy[:,2] - boxes.xyxy[:,0]) * (boxes.xyxy[:,3] - boxes.xyxy[:,1])
    idx = int(torch.argmax(areas).item())
    if res.keypoints is None or len(res.keypoints.data) == 0:
        return None
    if idx >= res.keypoints.data.shape[0]:
        return None
    return {
        "bbox": boxes.xyxy[idx].cpu().numpy(),          # [x1,y1,x2,y2]
        "kpts": res.keypoints.data[idx].cpu().numpy()   # (17,3)
    }

def bbox_aspect_ratio(b):
    w = float(b[2]-b[0]); h = float(b[3]-b[1])
    return (w / max(1e-6, h), w, h)

def torso_angle_deg(kpts):
    ls = get_joint(kpts, 5); rs = get_joint(kpts, 6)
    lh = get_joint(kpts, 11); rh = get_joint(kpts, 12)
    if ls and rs and lh and rh:
        mid_sh = ((ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0)
        mid_hp = ((lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0)
        return abs(angle(mid_sh, mid_hp)) % 180.0  # 0=horizontal, 90=vertical
    return None

def normalize_seq(seq):  # (T,17,3) -> (T,51)
    xy = seq[...,:2]
    mn = xy.min(axis=(0,1), keepdims=True)
    mx = xy.max(axis=(0,1), keepdims=True)
    xy = (xy - mn) / np.clip(mx - mn, 1e-6, None)
    out = np.concatenate([xy, seq[...,2:3]], axis=-1)
    return out.reshape(out.shape[0], -1)

# --------------------------- Optional LSTM classifier --------------------------
LSTM_MODEL = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if USE_CLASSIFIER:
    class LSTMFall(torch.nn.Module):
        def __init__(self, input_dim=51, hidden=128, layers=1, bidir=True, dropout=0.1):
            super().__init__()
            self.rnn = torch.nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True,
                                     bidirectional=bidir, dropout=dropout if layers>1 else 0.0)
            out_dim = hidden*(2 if bidir else 1)
            self.head = torch.nn.Sequential(
                torch.nn.LayerNorm(out_dim),
                torch.nn.Linear(out_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 2)
            )
        def forward(self, x):
            _, (hn, _) = self.rnn(x)
            h = torch.cat([hn[-1], hn[-2]], dim=-1) if self.rnn.bidirectional else hn[-1]
            return self.head(h)
    LSTM_MODEL = LSTMFall().to(DEVICE)
    LSTM_MODEL.load_state_dict(torch.load(CLS_WEIGHTS, map_location=DEVICE))
    LSTM_MODEL.eval()

# --------------------------- Event Logger --------------------------------------
class EventLogger:
    HEADERS = [
        "iso_time", "unix_ts", "camera_index", "status",
        "fall_score", "alert_thresh", "frame_w", "frame_h",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "snapshot_path"
    ]
    def __init__(self, csv_path=LOG_CSV, snap_dir=SNAP_DIR, cooldown_sec=COOLDOWN_SEC):
        self.csv_path = csv_path
        self.snap_dir = snap_dir
        self.cooldown_sec = cooldown_sec
        self.last_alert_ts = 0.0
        # Prepare CSV
        new_file = not os.path.exists(csv_path)
        if new_file:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.HEADERS)
        # Prepare snapshot folder
        if snap_dir:
            pathlib.Path(snap_dir).mkdir(parents=True, exist_ok=True)

    def should_log(self):
        now = time.time()
        if now - self.last_alert_ts >= self.cooldown_sec:
            self.last_alert_ts = now
            return True
        return False

    def save_snapshot(self, frame):
        if not self.snap_dir:
            return ""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(self.snap_dir, f"fall_{ts}.jpg")
        cv2.imwrite(path, frame)
        return path

    def log(self, cam_idx, status, score, thresh, frame_shape, bbox, snapshot_path=""):
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = (bbox if bbox is not None else [None]*4)
        row = [
            datetime.datetime.now().isoformat(timespec="seconds"),
            f"{time.time():.3f}",
            cam_idx, status, f"{score:.4f}", f"{thresh:.2f}",
            w, h, 
            x1, y1, x2, y2,
            snapshot_path
        ]
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

# --------------------------- Main live loop ------------------------------------
def main(camera_index=0, model_name="yolo11n-pose.pt"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    model = YOLO(model_name)
    logger = EventLogger(LOG_CSV, SNAP_DIR, COOLDOWN_SEC)

    buf_seq, center_y_hist = [], []
    prev_t, fps_ema = time.time(), 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=CONF_THRES, verbose=False)
        res = results[0]
        picked = largest_person_result(res)
        vis = res.plot()

        fall_score, status, bbox = 0.0, "OK", None

        if picked is not None:
            kpts = picked["kpts"]
            bbox = picked["bbox"]
            buf_seq.append(kpts.astype(np.float32))
            if len(buf_seq) > SEQ_LEN:
                buf_seq.pop(0)

            # Heuristics
            ar, w, h = bbox_aspect_ratio(bbox)
            wide_score = np.clip((ar - 0.9) / 0.6, 0.0, 1.0)

            t_angle = torso_angle_deg(kpts)
            if t_angle is not None:
                horiz = min(abs(t_angle - 0.0), abs(t_angle - 180.0))
                torso_score = np.clip((30.0 - horiz) / 30.0, 0.0, 1.0)
            else:
                torso_score = 0.0

            cx = (bbox[0]+bbox[2])/2.0
            cy = (bbox[1]+bbox[3])/2.0
            center_y_hist.append(cy)
            if len(center_y_hist) > 6:
                center_y_hist.pop(0)
            drop_score = 0.0
            if len(center_y_hist) >= 4:
                v1 = center_y_hist[-1] - center_y_hist[-2]
                v2 = center_y_hist[-2] - center_y_hist[-3]
                drop_pix = max(0.0, v1 + 0.5*v2)
                drop_score = np.clip(drop_pix / 40.0, 0.0, 1.0)

            shrink_score = 0.0  # kept simple for MVP; can add ankle-head span vs bbox height

            heur_score = 0.6*wide_score + 0.3*torso_score + 0.1*drop_score

            if USE_CLASSIFIER and len(buf_seq) == SEQ_LEN:
                x = normalize_seq(np.stack(buf_seq, 0))  # (T,51)
                x = torch.tensor(x[None,...], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    p = torch.softmax(LSTM_MODEL(x), dim=-1)[0].cpu().numpy()
                cls_score = float(p[1])
            else:
                cls_score = heur_score

            fall_score = 0.7*cls_score + 0.3*heur_score if USE_CLASSIFIER else heur_score
            if fall_score >= ALERT_THRESH and len(buf_seq) >= MIN_FRAMES_FOR_SCORE:
                status = "ALERT: FALL"

        # Draw UI
        color = (0,0,255) if status != "OK" else (0,255,0)
        cv2.putText(vis, f"Score: {fall_score:.2f}", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vis, status, (8, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)

        # FPS
        now = time.time()
        fps_ema = 0.9*fps_ema + 0.1*(1.0 / max(1e-6, now - prev_t))
        prev_t = now
        cv2.putText(vis, f"FPS: {fps_ema:.1f}", (8, vis.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        # -------------------- LOGGING on ALERT (with cooldown) --------------------
        if status == "ALERT: FALL" and logger.should_log():
            snap_path = logger.save_snapshot(vis) if SNAP_DIR else ""
            logger.log(camera_index, status, fall_score, ALERT_THRESH, vis.shape, bbox, snap_path)

        cv2.imshow("Live Fall Detection (YOLO-Pose)", vis)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------------- Entry point ---------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Webcam index (0/1/2...)")
    ap.add_argument("--model", default="yolo11n-pose.pt", help="yolo11n-pose.pt / yolo11s-pose.pt ...")
    ap.add_argument("--alert", type=float, default=ALERT_THRESH, help="alert threshold (0-1)")
    ap.add_argument("--cooldown", type=float, default=COOLDOWN_SEC, help="seconds between logs")
    ap.add_argument("--log_csv", default=LOG_CSV, help="CSV file for event logs")
    ap.add_argument("--snap_dir", default=SNAP_DIR, help="folder to save alert snapshots; '' to disable")
    args = ap.parse_args()

    # allow runtime override
    ALERT_THRESH = args.alert
    COOLDOWN_SEC = args.cooldown
    LOG_CSV = args.log_csv
    SNAP_DIR = args.snap_dir

    main(args.camera, args.model)
