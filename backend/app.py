import argparse
import base64
import collections
import json
import os
import threading
import time
import traceback

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

# ─── Optional LaneNet deep model (only used if weights exist) ───────────────
try:
    from lanenet_model import LaneNetModel
    _LANENET_AVAILABLE = True
except ImportError:
    _LANENET_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════
YOLO_DEVICE      = os.environ.get('YOLO_DEVICE', 'cpu')
YOLO_MODEL_PATH  = os.environ.get('YOLO_MODEL', 'yolov8s.pt')
LANENET_WEIGHTS  = os.environ.get('LANENET_WEIGHTS', 'models/lanenet.pth')
TRAIN_RESULTS_DIR = 'train_results'
os.makedirs(TRAIN_RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  Sign detector  (YOLOv8)
# ═══════════════════════════════════════════════════════════════

# ── Road-relevant COCO classes and their display colors (BGR) ──────────────
ROAD_CLASSES = {
    'person':         (0,   200, 255),  # orange-yellow  – pedestrian
    'bicycle':        (255, 200,   0),  # light blue
    'car':            (0,   0,   255),  # red
    'motorcycle':     (0,  128, 255),   # orange
    'bus':            (0,   60, 220),   # dark red
    'truck':          (0,   40, 180),   # darker red
    'traffic light':  (0,  255,   0),   # green
    'stop sign':      (0,   0,  200),   # darker red
    'fire hydrant':   (0, 165,  255),   # orange
    'parking meter':  (200, 200,   0),  # teal-ish
}
# COCO class ids for quick lookup
ROAD_CLASS_IDS = {0, 1, 2, 3, 5, 7, 9, 10, 11, 12}


class SignDetector:
    def __init__(self, model_name=YOLO_MODEL_PATH, device='cpu', conf=0.30):
        self.model  = YOLO(model_name)
        self.device = device
        self.conf   = conf
        self.names  = self.model.names

    def detect(self, frame):
        results = self.model(frame, device=self.device, verbose=False, conf=self.conf)
        if not results:
            return []
        boxes = getattr(results[0], 'boxes', None)
        if boxes is None:
            return []
        out = []
        for i in range(len(boxes)):
            c = float(boxes.conf[i])
            if c < self.conf:
                continue
            cls   = int(boxes.cls[i])
            # Only keep road-relevant classes
            if cls not in ROAD_CLASS_IDS:
                continue
            xyxy  = boxes.xyxy[i].cpu().numpy()
            label = self.names.get(cls, str(cls)) if isinstance(self.names, dict) else self.names[cls]
            x1, y1, x2, y2 = map(int, xyxy)
            color = list(ROAD_CLASSES.get(label, (0, 0, 255)))
            out.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'label': label, 'conf': c, 'color': color})
        return out


# ═══════════════════════════════════════════════════════════════
#  ▶  TRAINING  –  YOLOv8 fine-tuning
# ═══════════════════════════════════════════════════════════════

# Shared state for background training
_train_state = {
    'running':   False,
    'done':      False,
    'error':     None,
    'progress':  '',
    'results':   None,
}


def train_yolo(data_yaml: str,
               model_name: str = YOLO_MODEL_PATH,
               epochs: int = 50,
               imgsz: int = 640,
               batch: int = 16,
               project: str = TRAIN_RESULTS_DIR,
               name: str = 'yolo_finetune',
               device: str = YOLO_DEVICE):
    """
    Fine-tune YOLOv8 on a custom COCO-format dataset.

    Args:
        data_yaml:  Path to a YOLO data.yaml file.
        model_name: Base model weights (e.g. 'yolov8n.pt').
        epochs:     Number of training epochs.
        imgsz:      Input image size.
        batch:      Batch size.
        project:    Output directory root.
        name:       Run subdirectory name.
        device:     'cpu' | '0' | 'cuda'.

    Returns:
        dict with metrics and saved model path.
    """
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        exist_ok=True,
    )
    best_path = os.path.join(project, name, 'weights', 'best.pt')
    summary = {
        'best_model':    best_path,
        'results_dir':   os.path.join(project, name),
        'epochs_trained': epochs,
    }
    # Attempt to read final metrics from results.csv if present
    csv_path = os.path.join(project, name, 'results.csv')
    if os.path.exists(csv_path):
        try:
            import csv
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                summary['final_metrics'] = {k.strip(): v.strip() for k, v in last.items()}
        except Exception:
            pass
    return summary


def _train_yolo_bg(data_yaml, **kwargs):
    """Run train_yolo in a background thread; update _train_state."""
    global _train_state
    _train_state.update({'running': True, 'done': False, 'error': None,
                         'progress': 'Starting…', 'results': None})
    try:
        _train_state['progress'] = 'Training in progress…'
        results = train_yolo(data_yaml, **kwargs)
        _train_state['results']  = results
        _train_state['progress'] = 'Done'
    except Exception as e:
        _train_state['error']    = traceback.format_exc()
        _train_state['progress'] = f'Error: {e}'
    finally:
        _train_state['running']  = False
        _train_state['done']     = True


# ═══════════════════════════════════════════════════════════════
#  ▶  TESTING / EVALUATION  –  YOLOv8 val
# ═══════════════════════════════════════════════════════════════

def test_yolo(data_yaml: str,
              model_path: str = YOLO_MODEL_PATH,
              imgsz: int = 640,
              batch: int = 16,
              device: str = YOLO_DEVICE,
              split: str = 'val') -> dict:
    """
    Run YOLO validation and return mAP metrics.

    Args:
        data_yaml:  Path to data.yaml.
        model_path: Trained model weights (.pt file).
        imgsz:      Image size.
        batch:      Batch size.
        device:     Device string.
        split:      Dataset split to evaluate ('val' or 'test').

    Returns:
        dict with mAP50, mAP50-95, precision, recall, etc.
    """
    model   = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        split=split,
        verbose=True,
    )
    # ultralytics metrics object → plain dict
    results = {}
    for attr in ('box', 'seg', 'pose', 'obb'):
        m = getattr(metrics, attr, None)
        if m is None:
            continue
        try:
            results[attr] = {
                'map50':    float(m.map50),
                'map50_95': float(m.map),
                'mp':       float(m.mp),
                'mr':       float(m.mr),
            }
        except Exception:
            pass
    if not results:
        # Fallback – try generic attrs
        for k in ('map50', 'map50_95', 'map', 'mp', 'mr'):
            v = getattr(metrics, k, None)
            if v is not None:
                try:
                    results[k] = float(v)
                except Exception:
                    pass
    return results


# ═══════════════════════════════════════════════════════════════
#  ▶  LANE TRAINING  –  Simple segmentation fine-tuning
# ═══════════════════════════════════════════════════════════════

def train_lane_model(images_dir: str,
                     masks_dir: str,
                     weights_out: str = LANENET_WEIGHTS,
                     epochs: int = 30,
                     lr: float = 1e-4,
                     batch: int = 4,
                     imgsz: tuple = (512, 256),
                     device_str: str = 'cpu') -> dict:
    """
    Train a minimal U-Net lane segmentation model from scratch.

    Expects:
        images_dir/  *.jpg  (or *.png)   – BGR road images
        masks_dir/   *.png              – binary lane masks (255 = lane)

    The image filename stem must match its mask stem.
    Output: TorchScript model saved to weights_out.

    Returns: dict with training summary.
    """
    import glob
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.functional as TF

    # ── Tiny U-Net architecture ──────────────────────────────────
    class DoubleConv(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.net(x)

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1  = DoubleConv(3,   32)
            self.enc2  = DoubleConv(32,  64)
            self.enc3  = DoubleConv(64, 128)
            self.pool  = nn.MaxPool2d(2)
            self.up2   = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2  = DoubleConv(128, 64)
            self.up1   = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1  = DoubleConv(64,  32)
            self.head  = nn.Conv2d(32, 1, 1)
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(e3), e2], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            return self.head(d1)  # (B,1,H,W) logits

    # ── Dataset ──────────────────────────────────────────────────
    class LaneDataset(Dataset):
        def __init__(self, img_dir, msk_dir, size):
            self.size = size  # (W,H)
            stems = {os.path.splitext(os.path.basename(p))[0]: p
                     for p in glob.glob(os.path.join(img_dir, '*'))}
            self.pairs = []
            for mp in glob.glob(os.path.join(msk_dir, '*')):
                stem = os.path.splitext(os.path.basename(mp))[0]
                if stem in stems:
                    self.pairs.append((stems[stem], mp))
            if not self.pairs:
                raise RuntimeError(f"No matching image/mask pairs in {img_dir} & {msk_dir}")

        def __len__(self): return len(self.pairs)

        def __getitem__(self, idx):
            ip, mp = self.pairs[idx]
            img  = cv2.resize(cv2.imread(ip), self.size)[:, :, ::-1].copy()
            mask = cv2.resize(cv2.imread(mp, cv2.IMREAD_GRAYSCALE), self.size)
            img  = (img.astype(np.float32) / 255.0 - np.array([0.485,0.456,0.406])) \
                    / np.array([0.229,0.224,0.225])
            img  = torch.from_numpy(img.transpose(2,0,1)).float()
            mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
            return img, mask

    import torch
    device = torch.device(device_str if torch.cuda.is_available() or device_str == 'cpu' else 'cpu')
    ds     = LaneDataset(images_dir, masks_dir, imgsz)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model  = TinyUNet().to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        total = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            opt.zero_grad()
            pred = model(imgs)
            loss = loss_fn(pred, masks)
            loss.backward()
            opt.step()
            total += loss.item()
        avg = total / len(loader)
        history.append(avg)
        print(f"  [LaneTrain] Epoch {ep:3d}/{epochs}  loss={avg:.4f}")

    # Save as TorchScript
    model.eval()
    os.makedirs(os.path.dirname(weights_out) if os.path.dirname(weights_out) else '.', exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(weights_out)
    print(f"  [LaneTrain] Saved TorchScript model → {weights_out}")
    return {
        'weights_saved': weights_out,
        'epochs': epochs,
        'final_loss': history[-1] if history else None,
        'loss_history': history,
    }


# ═══════════════════════════════════════════════════════════════
#  ▶  LANE TESTING  –  IoU evaluation
# ═══════════════════════════════════════════════════════════════

def test_lane_model(images_dir: str,
                    masks_dir: str,
                    weights_path: str = LANENET_WEIGHTS,
                    imgsz: tuple = (512, 256),
                    prob_thresh: float = 0.5) -> dict:
    """
    Compute mean IoU for lane segmentation on a directory of images + masks.

    Returns:
        dict with mean_iou, pixel_accuracy, per_image results list.
    """
    import glob
    import torch
    if not _LANENET_AVAILABLE:
        raise RuntimeError("lanenet_model.py not found. Cannot run lane test.")
    if not os.path.exists(weights_path):
        raise RuntimeError(f"Weights not found at {weights_path}")

    lane_model = LaneNetModel(weight_path=weights_path, device='cpu', input_size=imgsz)

    stems = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(images_dir, '*'))}
    pairs = []
    for mp in glob.glob(os.path.join(masks_dir, '*')):
        stem = os.path.splitext(os.path.basename(mp))[0]
        if stem in stems:
            pairs.append((stems[stem], mp))

    if not pairs:
        raise RuntimeError("No matching image/mask pairs found.")

    ious, accs = [], []
    per_image  = []
    for ip, mp in pairs:
        img  = cv2.imread(ip)
        mask_gt = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if img is None:
            continue
        pred_mask = lane_model.predict_mask(img, prob_thresh=prob_thresh)
        pred_bin  = (pred_mask > 127).astype(np.uint8)
        pred_r    = cv2.resize(pred_bin, (mask_gt.shape[1], mask_gt.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        inter = int(np.logical_and(pred_r, mask_gt).sum())
        union = int(np.logical_or (pred_r, mask_gt).sum())
        iou   = inter / (union + 1e-6)
        acc   = float(np.mean(pred_r == mask_gt))
        ious.append(iou)
        accs.append(acc)
        per_image.append({
            'image': os.path.basename(ip),
            'iou':   round(iou,  4),
            'acc':   round(acc,  4),
        })

    return {
        'num_images':      len(pairs),
        'mean_iou':        round(float(np.mean(ious))  if ious else 0, 4),
        'pixel_accuracy':  round(float(np.mean(accs))  if accs else 0, 4),
        'per_image':       per_image,
    }


# ═══════════════════════════════════════════════════════════════
#  Road existence gate
# ═══════════════════════════════════════════════════════════════

def road_confidence(frame) -> float:
    """
    Returns a score 0-95.  >= 40 → road present.
    Uses 4 independent signals so a single false positive can't pass.
    """
    fh, fw = frame.shape[:2]
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = 0.0

    # ── Signal 1: Largest contiguous asphalt-coloured region ──────────────────
    bot_hsv = hsv[int(fh * 0.55):, :]
    asp     = cv2.inRange(bot_hsv,
                          np.array([0,   0,  30]),
                          np.array([180, 80, 185]))
    n, _, stats, _ = cv2.connectedComponentsWithStats(asp, connectivity=8)
    if n > 1:
        big = int(stats[1:, cv2.CC_STAT_AREA].max())
        if big / (bot_hsv.shape[0] * fw + 1) > 0.12:
            score += 25

    # ── Signal 2: BALANCED converging diagonal lines ───────────────────────────
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=60, minLineLength=60, maxLineGap=20)
    ld = rd = 0
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(x2 - x1) < 1:
                continue
            s = (y2 - y1) / (x2 - x1)
            if  0.3 < s < 5:    ld += 1
            if -5   < s < -0.3: rd += 1
    if ld >= 2 and rd >= 2:
        balance = min(ld, rd) / (max(ld, rd) + 1)
        if balance > 0.05:
            score += 30

    # ── Signal 3: Lane-coloured vertical streaks ───────────────────────────────
    h2_hsv = hsv[fh // 2:, :]
    white  = cv2.inRange(h2_hsv, np.array([0,   0, 200]), np.array([180, 40, 255]))
    yellow = cv2.inRange(h2_hsv, np.array([15, 70,  70]), np.array([42, 255, 255]))
    lm     = cv2.bitwise_or(white, yellow)
    lp     = cv2.countNonZero(lm) / (fh // 2 * fw + 1) * 100
    col_pct     = np.sum(lm > 0, axis=0).astype(float) / (fh // 2) * 100
    streak_cols = int(np.sum(col_pct > 4))
    if 1.0 < lp < 45 and 2 < streak_cols < fw * 0.45:
        score += 25

    # ── Signal 4: Medium texture variance ─────────────────────────────────────
    g_std = float(cv2.meanStdDev(gray[int(fh * 0.55):, :].astype(float))[1][0][0])
    if 6 < g_std < 62:
        score += 15

    return score


def _no_road_overlay(frame, msg="No road detected"):
    out = frame.copy()
    overlay = out.copy()
    h, w = out.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    font  = cv2.FONT_HERSHEY_DUPLEX
    lines = [msg, "Point camera at a road to begin detection."]
    y0    = h // 2 - 30
    for i, line in enumerate(lines):
        sz, _ = cv2.getTextSize(line, font, 0.85, 2)
        tx = (w - sz[0]) // 2
        ty = y0 + i * 48
        cv2.putText(out, line, (tx, ty), font, 0.85, (0, 0, 0),   4, cv2.LINE_AA)
        cv2.putText(out, line, (tx, ty), font, 0.85, (0, 200, 255), 2, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════
#  Lane Detection Internals
# ═══════════════════════════════════════════════════════════════
_N    = 8
_lbuf = collections.deque(maxlen=_N)
_rbuf = collections.deque(maxlen=_N)


def _preprocess(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def _lane_color_mask(bgr):
    """WHITE + YELLOW lane mask; suppresses red/pink crosswalk paint."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS)

    # White: high value, low saturation
    white = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   0, 185]), np.array([180, 45, 255])),
        cv2.inRange(hls, np.array([0, 190,   0]), np.array([180, 255, 50]))
    )
    # Yellow: hue 15-38, reasonable saturation & value
    yellow = cv2.inRange(hsv, np.array([15, 70, 90]), np.array([38, 255, 255]))
    color  = cv2.bitwise_or(white, yellow)

    # Suppress red/pink crosswalk paint (hue near 0 or 170-180, high saturation)
    red1 = cv2.inRange(hsv, np.array([0,   80,  60]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 80,  60]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    # Dilate red mask slightly to cover reddish areas fully
    kr = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    red_mask = cv2.dilate(red_mask, kr)
    color = cv2.bitwise_and(color, cv2.bitwise_not(red_mask))

    gray  = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    sx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    asx   = (np.abs(sx) / (np.abs(sx).max() + 1e-6) * 255).astype(np.uint8)
    grad  = np.zeros_like(asx)
    grad[(asx >= 20) & (asx <= 220)] = 255

    if cv2.countNonZero(color) > 200:
        combined = cv2.bitwise_or(color, cv2.bitwise_and(grad, color))
    else:
        combined = grad

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k)
    return combined


def _detect_horizon(bgr):
    fh, fw = bgr.shape[:2]
    gray   = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (11, 11), 0)
    sy     = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    energy = np.sum(sy, axis=1)
    y_lo, y_hi = int(fh * 0.20), int(fh * 0.65)
    row    = int(np.argmax(energy[y_lo:y_hi])) + y_lo
    return int(np.clip(row, fh * 0.25, fh * 0.60))


def _roi_mask(binary, h, w, horizon):
    """Tighter trapezoid ROI: excludes sidewalks and far-edge noise."""
    mask = np.zeros_like(binary)
    pts  = np.array([[
        (int(w * 0.05), h - 1),         # bottom-left (slightly inset)
        (int(w * 0.30), horizon + 10),   # near-horizon left (narrower)
        (int(w * 0.70), horizon + 10),   # near-horizon right (narrower)
        (int(w * 0.95), h - 1),          # bottom-right (slightly inset)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(binary, mask)


def _perspective_matrices(h, w, horizon):
    src = np.float32([
        [int(w * 0.40), horizon],
        [int(w * 0.60), horizon],
        [int(w * 0.94), h - 8],
        [int(w * 0.06), h - 8],
    ])
    dst = np.float32([
        [w * 0.20, 0],
        [w * 0.80, 0],
        [w * 0.80, h],
        [w * 0.20, h],
    ])
    M    = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def _hough_bases(binary_roi, h, w, M):
    roi   = binary_roi[h // 2:, :]
    lines = cv2.HoughLinesP(roi, 1, np.pi / 180,
                             threshold=35, minLineLength=35, maxLineGap=20)
    if lines is None:
        return None, None
    lxs, rxs = [], []
    cx = w / 2.0
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if abs(x2 - x1) < 1:
            continue
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.25:
            continue
        mx = (x1 + x2) / 2.0
        (lxs if mx < cx else rxs).append(mx)

    def warp_x(ox):
        pt = cv2.perspectiveTransform(
            np.array([[[float(ox), float(h - 1)]]], dtype=np.float32), M)
        return int(np.clip(pt[0, 0, 0], 0, w - 1))

    lb = warp_x(int(np.mean(lxs))) if lxs else None
    rb = warp_x(int(np.mean(rxs))) if rxs else None
    return lb, rb


def _sliding_window(warped, hlb=None, hrb=None, nwindows=10, margin=90, minpix=35):
    H, W  = warped.shape
    hist  = np.sum(warped[H * 2 // 3:, :], axis=0).astype(np.float32)
    mid   = W // 2
    lb    = int(np.argmax(hist[:mid]))
    rb    = int(np.argmax(hist[mid:]) + mid)
    if hlb is not None and hist[lb] < 250:
        lb = int(np.clip(hlb, 0, mid - 1))
    if hrb is not None and hist[rb] < 250:
        rb = int(np.clip(hrb, mid, W - 1))

    nz       = warped.nonzero()
    nzy, nzx = np.array(nz[0]), np.array(nz[1])
    win_h    = H // nwindows
    lcur, rcur = lb, rb
    linds, rinds = [], []

    for wi in range(nwindows):
        y_lo = H - (wi + 1) * win_h
        y_hi = H - wi * win_h
        gl = np.where((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= lcur - margin) & (nzx < lcur + margin))[0]
        gr = np.where((nzy >= y_lo) & (nzy < y_hi) &
                      (nzx >= rcur - margin) & (nzx < rcur + margin))[0]
        if gl.size: linds.append(gl)
        if gr.size: rinds.append(gr)
        if gl.size > minpix: lcur = int(np.mean(nzx[gl]))
        if gr.size > minpix: rcur = int(np.mean(nzx[gr]))

    lfit = rfit = None
    lc   = rc   = 0
    if linds:
        li = np.concatenate(linds); lc = li.size
        if lc >= 60:
            lfit = np.polyfit(nzy[li], nzx[li], 2)
    if rinds:
        ri = np.concatenate(rinds); rc = ri.size
        if rc >= 60:
            rfit = np.polyfit(nzy[ri], nzx[ri], 2)
    return lfit, rfit, lc, rc


def _sanity(lfit, rfit, h, w):
    if lfit is None or rfit is None:
        return True
    ploty  = np.linspace(0, h - 1, 20)
    widths = np.polyval(rfit, ploty) - np.polyval(lfit, ploty)
    return (widths.min() > 80) and (widths.max() < w * 0.88) and (widths.std() < 160)


def _mirror(known, ploty, lane_w, right_known):
    a, b, c = known
    kx    = a * ploty**2 + b * ploty + c
    shift = lane_w / np.sqrt(1.0 + (2*a*ploty + b)**2)
    mx    = kx - shift if right_known else kx + shift
    return np.polyfit(ploty, mx, 2)


def _smooth(buf):
    if not buf:
        return None
    w = np.arange(1, len(buf) + 1, dtype=np.float32)
    w /= w.sum()
    return np.average(np.stack(list(buf)), axis=0, weights=w)


def _curvature(fit, h, xm, ym):
    A = fit[0] * (xm / ym**2)
    B = fit[1] * (xm / ym)
    d = (1 + (2*A*(h-1)*ym + B)**2) ** 1.5
    n = abs(2*A)
    return d / n if n > 1e-6 else 9999.0


# ═══════════════════════════════════════════════════════════════
#  Main detect_lanes entry point
# ═══════════════════════════════════════════════════════════════

def detect_lanes(frame):
    """
    Returns (annotated_frame, road_found:bool).
    If no road is detected, returns the frame with a warning overlay.
    """
    global _lbuf, _rbuf
    h, w = frame.shape[:2]

    # ── GATE: is there actually a road? ─────────────────────────
    rscore = road_confidence(frame)
    if rscore < 30:   # Lowered from 40 – prevents false negatives on bright/colored roads
        _lbuf.clear()
        _rbuf.clear()
        return _no_road_overlay(frame), False

    # ── Pre-process ──────────────────────────────────────────────
    enhanced = _preprocess(frame)

    # ── Binary mask (white + yellow only) ────────────────────────
    binary   = _lane_color_mask(enhanced)

    # ── Adaptive horizon ─────────────────────────────────────────
    horizon  = _detect_horizon(enhanced)

    # ── ROI ──────────────────────────────────────────────────────
    binary   = _roi_mask(binary, h, w, horizon)

    # ── Bird's-eye warp ──────────────────────────────────────────
    M, Minv  = _perspective_matrices(h, w, horizon)
    warped   = cv2.warpPerspective(binary, M, (w, h), flags=cv2.INTER_LINEAR)

    # ── Hough seeds ──────────────────────────────────────────────
    hlb, hrb = _hough_bases(binary, h, w, M)

    # ── Sliding window ───────────────────────────────────────────
    lfit, rfit, lc, rc = _sliding_window(warped, hlb, hrb)

    # ── Sanity check ─────────────────────────────────────────────
    if not _sanity(lfit, rfit, h, w):
        lfit = None if lc < rc else lfit
        rfit = None if rc < lc else rfit

    ploty  = np.linspace(0, h - 1, h)

    # ── Lane width estimate ──────────────────────────────────────
    if lfit is not None and rfit is not None:
        lane_w = int(np.clip(
            abs(np.polyval(rfit, h-1) - np.polyval(lfit, h-1)), 150, int(w*0.72)))
    else:
        lane_w = int(np.clip(w * 0.34, 150, int(w * 0.72)))

    # ── Mirror missing side ──────────────────────────────────────
    if   lfit is None and rfit is not None:
        lfit = _mirror(rfit, ploty, lane_w, right_known=True)
    elif rfit is None and lfit is not None:
        rfit = _mirror(lfit, ploty, lane_w, right_known=False)
    elif lfit is None and rfit is None:
        out = frame.copy()
        cv2.putText(out, f"Road detected (score:{rscore:.0f}) — lane lines not visible",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.70,
                    (0, 0, 0),     3, cv2.LINE_AA)
        cv2.putText(out, f"Road detected (score:{rscore:.0f}) — lane lines not visible",
                    (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.70,
                    (0, 200, 255), 1, cv2.LINE_AA)
        return out, True

    # ── Temporal smoothing ───────────────────────────────────────
    _lbuf.append(lfit); _rbuf.append(rfit)
    lf = _smooth(_lbuf)
    rf = _smooth(_rbuf)

    lfx  = np.polyval(lf, ploty)
    rfx  = np.polyval(rf, ploty)
    cx   = (lfx + rfx) / 2.0

    # ── Metrics ──────────────────────────────────────────────────
    xm       = 3.7 / float(lane_w)
    ym       = 30.0 / h
    offset_m = (w / 2.0 - cx[-1]) * xm
    curv     = (_curvature(lf, h, xm, ym) + _curvature(rf, h, xm, ym)) / 2.0
    abs_off  = abs(offset_m)

    fill_col = (0, 210, 0) if abs_off < 0.3 else \
               (0, 140, 255) if abs_off < 0.7 else (0, 0, 220)

    # ── Draw ─────────────────────────────────────────────────────
    cw   = np.zeros_like(frame)
    pl   = np.array([np.transpose(np.vstack([lfx, ploty]))], dtype=np.int32)
    pr   = np.array([np.flipud(np.transpose(np.vstack([rfx, ploty])))], dtype=np.int32)
    cv2.fillPoly(cw, np.int_([np.hstack((pl, pr))]), fill_col)
    cv2.polylines(cw, [pl.reshape(-1, 2)], False, (0, 220, 255), 7)
    cv2.polylines(cw, [pr.reshape(-1, 2)], False, (0, 220, 255), 7)
    cp = np.int32(np.transpose(np.vstack([cx, ploty]))).reshape(-1, 1, 2)
    cv2.polylines(cw, [cp], False, (220, 0, 200), 4)

    result = cv2.addWeighted(frame, 0.75,
                             cv2.warpPerspective(cw, Minv, (w, h)), 0.55, 0)

    # ── HUD ──────────────────────────────────────────────────────
    def hud(img, text, y, col=(255, 255, 255)):
        cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_DUPLEX,
                    0.72, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (18, y), cv2.FONT_HERSHEY_DUPLEX,
                    0.72, col,   1, cv2.LINE_AA)

    side    = "left" if offset_m > 0 else "right"
    c_str   = f"{curv:.0f} m" if curv < 6000 else "straight"
    off_col = (0,210,0) if abs_off < 0.3 else (0,140,255) if abs_off < 0.7 else (0,80,255)
    hud(result, f"Offset  : {abs_off:.2f} m {side}", 40,  off_col)
    hud(result, f"Curve   : {c_str}",                76)
    hud(result, f"Lane W  : {lane_w} px",           112)
    hud(result, f"Road    : {rscore:.0f}/95",       148,  (120, 255, 120))

    # Steering arrow
    acx, acy = w // 2, int(h * 0.10)
    dx = int(np.clip((w/2.0 - cx[-1]) * 0.35, -w*0.18, w*0.18))
    cv2.arrowedLine(result, (acx, acy), (acx + dx, acy), (0, 255, 255), 5, tipLength=0.4)

    return result, True


# ═══════════════════════════════════════════════════════════════
#  Flask App
# ═══════════════════════════════════════════════════════════════
app          = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)
sign_detector = SignDetector(model_name=YOLO_MODEL_PATH, device=YOLO_DEVICE)

# Lazy-load LaneNet if weights exist
_lane_deep: 'LaneNetModel | None' = None

def _get_lane_deep():
    global _lane_deep
    if _lane_deep is None and _LANENET_AVAILABLE and os.path.exists(LANENET_WEIGHTS):
        try:
            _lane_deep = LaneNetModel(weight_path=LANENET_WEIGHTS, device='cpu')
            print(f"[LaneNet] Deep model loaded from {LANENET_WEIGHTS}")
        except Exception as e:
            print(f"[LaneNet] Could not load weights: {e}")
    return _lane_deep


# ─── Serve routes ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok',
                    'lanenet_available': _LANENET_AVAILABLE,
                    'lanenet_weights':   os.path.exists(LANENET_WEIGHTS)})


@app.route('/process_frame', methods=['POST'])
def process_frame():
    t0 = time.time()
    if 'frame' not in request.files:
        return jsonify({'error': 'no frame'}), 400
    img = cv2.imdecode(
        np.frombuffer(request.files['frame'].read(), np.uint8),
        cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'invalid image'}), 400

    out, road_found = detect_lanes(img)

    # Optionally overlay deep LaneNet mask if model loaded
    ld = _get_lane_deep()
    if ld is not None and road_found:
        try:
            from postprocess import mask_to_lane_polylines, draw_lanes_on_image
            mask = ld.predict_mask(img)
            polys = mask_to_lane_polylines(mask)
            out   = draw_lanes_on_image(out, polys, color=(255, 100, 0), thickness=4)
        except Exception as e:
            print(f"[LaneNet inference error] {e}")

    # Detect on full image (road or not) to catch objects even when road not found
    signs = sign_detector.detect(img)
    for s in signs:
        color = tuple(s.get('color', [0, 0, 255]))
        x1, y1, x2, y2 = s['x1'], s['y1'], s['x2'], s['y2']
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label_text = f"{s['label']} {s['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ly - th - 4), (x1 + tw + 4, ly + 2), color, -1)
        cv2.putText(out, label_text, (x1 + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(out, label_text, (x1 + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    _, jpeg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jsonify({
        'image':           base64.b64encode(jpeg.tobytes()).decode(),
        'signs':           signs,
        'road_detected':   road_found,
        'processing_time': time.time() - t0
    })


@app.route('/debug_mask', methods=['POST'])
def debug_mask():
    if 'frame' not in request.files:
        return jsonify({'error': 'no frame'}), 400
    img = cv2.imdecode(
        np.frombuffer(request.files['frame'].read(), np.uint8),
        cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'invalid image'}), 400

    h, w    = img.shape[:2]
    score   = road_confidence(img)
    enh     = _preprocess(img)
    horizon = _detect_horizon(enh)
    binary  = _lane_color_mask(enh)
    binary  = _roi_mask(binary, h, w, horizon)
    M, _    = _perspective_matrices(h, w, horizon)
    warped  = cv2.warpPerspective(binary, M, (w, h))
    overlay = cv2.addWeighted(img, 0.6,
                              cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), 0.7, 0)
    cv2.line(overlay, (0, horizon), (w, horizon), (0, 0, 255), 2)
    cv2.putText(overlay, f"Road score: {score:.0f}/95  horizon:{horizon}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def enc(x):
        return base64.b64encode(cv2.imencode('.jpg', x)[1].tobytes()).decode()

    return jsonify({
        'road_score': score,
        'horizon':    horizon,
        'mask':       enc(binary),
        'overlay':    enc(overlay),
        'warped':     enc(warped),
    })


# ─── Training API routes ─────────────────────────────────────────────────────

@app.route('/train', methods=['POST'])
def api_train():
    """
    Start a background YOLOv8 fine-tuning job.

    JSON body (all optional except data):
    {
        "data":    "path/to/data.yaml",   ← REQUIRED
        "model":   "yolov8n.pt",
        "epochs":  50,
        "imgsz":   640,
        "batch":   16,
        "name":    "my_run"
    }
    """
    if _train_state['running']:
        return jsonify({'error': 'Training already in progress'}), 409

    body   = request.get_json(force=True, silent=True) or {}
    data   = body.get('data')
    if not data:
        return jsonify({'error': '"data" (path to data.yaml) is required'}), 400
    if not os.path.exists(data):
        return jsonify({'error': f'data.yaml not found: {data}'}), 400

    kwargs = {
        'model_name': body.get('model',  YOLO_MODEL_PATH),
        'epochs':     int(body.get('epochs', 50)),
        'imgsz':      int(body.get('imgsz',  640)),
        'batch':      int(body.get('batch',  16)),
        'name':       body.get('name', 'yolo_finetune'),
        'device':     YOLO_DEVICE,
    }
    t = threading.Thread(target=_train_yolo_bg, args=(data,), kwargs=kwargs, daemon=True)
    t.start()
    return jsonify({'status': 'started', 'message': 'Training started in background. Poll /train_status.'})


@app.route('/train_status', methods=['GET'])
def api_train_status():
    """Poll background training progress."""
    return jsonify({
        'running':  _train_state['running'],
        'done':     _train_state['done'],
        'progress': _train_state['progress'],
        'error':    _train_state['error'],
        'results':  _train_state['results'],
    })


@app.route('/test', methods=['POST'])
def api_test():
    """
    Run YOLO validation synchronously and return metrics.

    JSON body:
    {
        "data":   "path/to/data.yaml",   ← REQUIRED
        "model":  "path/to/best.pt",
        "imgsz":  640,
        "batch":  16,
        "split":  "val"
    }
    """
    body   = request.get_json(force=True, silent=True) or {}
    data   = body.get('data')
    if not data:
        return jsonify({'error': '"data" (path to data.yaml) is required'}), 400
    if not os.path.exists(data):
        return jsonify({'error': f'data.yaml not found: {data}'}), 400

    model_path = body.get('model', YOLO_MODEL_PATH)
    try:
        results = test_yolo(
            data_yaml=data,
            model_path=model_path,
            imgsz=int(body.get('imgsz', 640)),
            batch=int(body.get('batch', 16)),
            split=body.get('split', 'val'),
        )
        return jsonify({'status': 'ok', 'metrics': results})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/train_lane', methods=['POST'])
def api_train_lane():
    """
    Train the LaneNet U-Net segmentation model.

    JSON body:
    {
        "images_dir": "path/to/images",   ← REQUIRED
        "masks_dir":  "path/to/masks",    ← REQUIRED
        "epochs":     30,
        "lr":         1e-4,
        "batch":      4,
        "weights_out": "models/lanenet.pth"
    }
    """
    body       = request.get_json(force=True, silent=True) or {}
    images_dir = body.get('images_dir')
    masks_dir  = body.get('masks_dir')
    if not images_dir or not masks_dir:
        return jsonify({'error': '"images_dir" and "masks_dir" are required'}), 400
    for d in (images_dir, masks_dir):
        if not os.path.isdir(d):
            return jsonify({'error': f'Directory not found: {d}'}), 400
    try:
        result = train_lane_model(
            images_dir  = images_dir,
            masks_dir   = masks_dir,
            weights_out = body.get('weights_out', LANENET_WEIGHTS),
            epochs      = int(body.get('epochs', 30)),
            lr          = float(body.get('lr', 1e-4)),
            batch       = int(body.get('batch', 4)),
        )
        # Reload the deep model
        global _lane_deep
        _lane_deep = None
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/test_lane', methods=['POST'])
def api_test_lane():
    """
    Evaluate LaneNet segmentation (IoU) on a directory of images + masks.

    JSON body:
    {
        "images_dir":   "path/to/images",   ← REQUIRED
        "masks_dir":    "path/to/masks",    ← REQUIRED
        "weights_path": "models/lanenet.pth",
        "prob_thresh":  0.5
    }
    """
    body       = request.get_json(force=True, silent=True) or {}
    images_dir = body.get('images_dir')
    masks_dir  = body.get('masks_dir')
    if not images_dir or not masks_dir:
        return jsonify({'error': '"images_dir" and "masks_dir" are required'}), 400
    try:
        result = test_lane_model(
            images_dir   = images_dir,
            masks_dir    = masks_dir,
            weights_path = body.get('weights_path', LANENET_WEIGHTS),
            prob_thresh  = float(body.get('prob_thresh', 0.5)),
        )
        return jsonify({'status': 'ok', 'result': result})
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════
#  CLI entry-point
# ═══════════════════════════════════════════════════════════════

def _cli_train(args):
    print("=" * 60)
    print(f"[CLI TRAIN]  data={args.data}  epochs={args.epochs}  model={args.model}")
    print("=" * 60)
    result = train_yolo(
        data_yaml  = args.data,
        model_name = args.model,
        epochs     = args.epochs,
        imgsz      = args.imgsz,
        batch      = args.batch,
        name       = args.name,
        device     = YOLO_DEVICE,
    )
    print("\n[TRAIN COMPLETE]")
    print(json.dumps(result, indent=2))


def _cli_test(args):
    print("=" * 60)
    print(f"[CLI TEST]  data={args.data}  model={args.model}")
    print("=" * 60)
    result = test_yolo(
        data_yaml  = args.data,
        model_path = args.model,
        imgsz      = args.imgsz,
        batch      = args.batch,
        split      = args.split,
    )
    print("\n[TEST RESULTS]")
    print(json.dumps(result, indent=2))


def _cli_train_lane(args):
    print("=" * 60)
    print(f"[CLI TRAIN_LANE]  images={args.images}  masks={args.masks}")
    print("=" * 60)
    result = train_lane_model(
        images_dir  = args.images,
        masks_dir   = args.masks,
        weights_out = args.weights_out,
        epochs      = args.epochs,
        lr          = args.lr,
        batch       = args.batch,
    )
    print("\n[LANE TRAIN COMPLETE]")
    print(json.dumps(result, indent=2))


def _cli_test_lane(args):
    print("=" * 60)
    print(f"[CLI TEST_LANE]  images={args.images}  masks={args.masks}")
    print("=" * 60)
    result = test_lane_model(
        images_dir   = args.images,
        masks_dir    = args.masks,
        weights_path = args.weights_path,
        prob_thresh  = args.prob_thresh,
    )
    print("\n[LANE TEST RESULTS]")
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane + Sign Detector server / trainer')
    parser.add_argument('--mode', choices=['serve', 'train', 'test', 'train_lane', 'test_lane'],
                        default='serve',
                        help='serve (default) | train | test | train_lane | test_lane')
    parser.add_argument('--port',   type=int, default=int(os.environ.get('PORT', 5000)))

    # YOLO train / test
    parser.add_argument('--data',   type=str, default=None,
                        help='Path to data.yaml (required for train / test)')
    parser.add_argument('--model',  type=str, default='yolov8s.pt',
                        help='YOLOv8 model name or path')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz',  type=int, default=640)
    parser.add_argument('--batch',  type=int, default=16)
    parser.add_argument('--name',   type=str, default='yolo_finetune')
    parser.add_argument('--split',  type=str, default='val')

    # LaneNet train / test
    parser.add_argument('--images',      type=str, default=None, help='Images directory for lane model')
    parser.add_argument('--masks',       type=str, default=None, help='Masks directory for lane model')
    parser.add_argument('--weights_out', type=str, default=LANENET_WEIGHTS,
                        help='Output path for trained lane model')
    parser.add_argument('--weights_path',type=str, default=LANENET_WEIGHTS,
                        help='Weights path for lane model evaluation')
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--prob_thresh', type=float, default=0.5)

    args = parser.parse_args()

    if args.mode == 'serve':
        print(f"Starting Lane+Sign Detector on http://0.0.0.0:{args.port}  (YOLO={YOLO_DEVICE})")
        app.run(host='0.0.0.0', port=args.port, debug=False)

    elif args.mode == 'train':
        if not args.data:
            parser.error('--data is required for --mode train')
        _cli_train(args)

    elif args.mode == 'test':
        if not args.data:
            parser.error('--data is required for --mode test')
        _cli_test(args)

    elif args.mode == 'train_lane':
        if not args.images or not args.masks:
            parser.error('--images and --masks are required for --mode train_lane')
        _cli_train_lane(args)

    elif args.mode == 'test_lane':
        if not args.images or not args.masks:
            parser.error('--images and --masks are required for --mode test_lane')
        _cli_test_lane(args)
