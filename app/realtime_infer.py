import argparse
import threading
import time
import queue
import sys
import os
from collections import deque

import numpy as np
import torch
import yaml

import cv2

# Optional imports for audio
try:
    import sounddevice as sd
    HAVE_SD = True
except Exception:
    sd = None
    HAVE_SD = False

# Add repo root to path
import pathlib as _pl
_repo_root = str(_pl.Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
# Local model imports
from models.GFMamba import GFMamba
# --- Camera selection helpers (prefer physical USB cam on Windows) ---
import platform as _pf

def _cap_api_from_name(name:str):
    name = (name or 'auto').lower()
    if name == 'dshow':
        return cv2.CAP_DSHOW
    if name == 'msmf':
        return cv2.CAP_MSMF
    return None  # auto/any -> let OpenCV choose


def open_camera_by_index(idx:int, backend:str='dshow'):
    api = _cap_api_from_name(backend)
    cap = cv2.VideoCapture(idx, api) if api is not None else cv2.VideoCapture(idx)
    if not cap.isOpened():
        try: cap.release()
        except Exception: pass
        return None
    ok, frame = cap.read()
    if not ok or frame is None or getattr(frame, 'size', 0) == 0:
        try: cap.release()
        except Exception: pass
        return None
    return cap


def open_usb_camera(preferred_index:int=-1, backend:str='dshow'):
    # If user specified an index, try it first
    if preferred_index is not None and preferred_index >= 0:
        cap = open_camera_by_index(preferred_index, backend)
        if cap is not None:
            return cap
    # Heuristic: on Windows, virtual cams (e.g., Iriun) often claim index 0.
    # Try non-zero indices first.
    order = [1, 2, 0, 3, 4, 5]
    tried = []
    for i in order:
        cap = open_camera_by_index(i, backend)
        if cap is not None:
            return cap
        tried.append(i)
    return None


def list_cameras(max_index=10, backend='dshow'):
    avail = []
    for i in range(max_index+1):
        cap = open_camera_by_index(i, backend)
        if cap is not None:
            avail.append(i)
            try: cap.release()
            except Exception: pass
    return avail



def load_config(cfg_path):
    with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as f:
        return yaml.safe_load(f)



def load_model(cfg_path, ckpt_path, device='cpu'):
    """Load config + model and restore checkpoint, robust to common wrappers.
    - Accepts keys: state_dict, model_state_dict, model
    - Strips DataParallel prefixes (module.)
    - Falls back to CPU map_location if CUDA not available
    """
    args = load_config(cfg_path)
    model = GFMamba(args)

    # Normalize device and choose safe map_location
    dev = device if isinstance(device, torch.device) else torch.device(device)
    want_cuda = (dev.type == 'cuda')
    if want_cuda and not torch.cuda.is_available():
        print('WARNING: CUDA not available. Loading checkpoint on CPU; set --device cpu or install a CUDA build of PyTorch.')
        dev = torch.device('cpu')
    map_loc = dev

    raw = torch.load(ckpt_path, map_location=map_loc, weights_only=False)

    # Unwrap common checkpoint formats
    state = raw
    if isinstance(state, dict):
        for k in ('state_dict', 'model_state_dict', 'model'):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    # Strip common prefixes
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith('module.'):
                nk = nk[7:]
            new_state[nk] = v
        state = new_state

    # Detect fusion head type from checkpoint
    has_linear_head = any(k.startswith('fusion_linear.') for k in state.keys())
    if hasattr(model, 'use_linear_head'):
        model.use_linear_head = bool(has_linear_head)
        if model.use_linear_head:
            print('Using compatibility fusion head: fusion_linear (from checkpoint)')

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Filter reporting for alternate head to reduce noise
    def _filter_keys(keys):
        out = []
        for k in list(keys):
            if getattr(model, 'use_linear_head', False):
                if k.startswith('graph_fusion.'):
                    continue
            else:
                if k.startswith('fusion_linear.'):
                    continue
            out.append(k)
        return out

    missing = _filter_keys(missing)
    unexpected = _filter_keys(unexpected)
    if missing:
        print(f'load_model: missing keys: {missing[:5]}... (+{max(0, len(missing)-5)} more)')
    if unexpected:
        print(f'load_model: unexpected keys: {unexpected[:5]}... (+{max(0, len(unexpected)-5)} more)')

    model.to(dev)
    model.eval()
    return model, args

class RingBuffer:
    def __init__(self, maxlen):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, x):
        with self.lock:
            self.buf.append(x)

    def get_all(self):
        with self.lock:
            return list(self.buf)

    def clear(self):
        with self.lock:
            self.buf.clear()


class AudioFeatureExtractor:
    """Compute simple 5-dim features per short-time frame using numpy only.
    Features per frame (~20 ms):
    - log energy
    - zero crossing rate
    - spectral centroid
    - spectral bandwidth
    - spectral rolloff (0.85)
    """
    def __init__(self, sample_rate=16000, frame_ms=20, hop_ms=20):
        self.fs = sample_rate
        self.frame = int(sample_rate * frame_ms / 1000)
        self.hop = int(sample_rate * hop_ms / 1000)
        self.window = np.hanning(self.frame).astype(np.float32)
        # Precompute freq bins
        self.freqs = np.fft.rfftfreq(self.frame, d=1.0 / self.fs).astype(np.float32)

    def frame_signal(self, x):
        # Returns frames [T, N]
        if len(x) < self.frame:
            x = np.pad(x, (0, self.frame - len(x)))
        n_frames = 1 + max(0, (len(x) - self.frame) // self.hop)
        frames = []
        for i in range(n_frames):
            start = i * self.hop
            seg = x[start:start + self.frame]
            if len(seg) < self.frame:
                seg = np.pad(seg, (0, self.frame - len(seg)))
            frames.append(seg)
        if not frames:
            frames = [np.pad(x, (0, self.frame - len(x)))[:self.frame]]
        return np.stack(frames, axis=0)

    def features(self, x):
        # x: mono int16/float32 -> float32
        x = np.asarray(x).astype(np.float32)
        if x.ndim > 1:
            x = x[:, 0]
        frames = self.frame_signal(x)
        # Window and FFT
        windowed = frames * self.window[None, :]
        mag = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-8
        pow_spec = (mag ** 2).astype(np.float32)

        # Energy
        energy = (windowed ** 2).sum(axis=1) + 1e-8
        log_energy = np.log(energy)

        # ZCR
        zcr = ((np.diff(np.sign(windowed), axis=1) != 0).sum(axis=1) / float(self.frame)).astype(np.float32)

        # Spectral centroid
        norm_mag = mag / mag.sum(axis=1, keepdims=True)
        centroid = (norm_mag * self.freqs[None, :]).sum(axis=1)

        # Spectral bandwidth (2nd central moment)
        bw = np.sqrt(((self.freqs[None, :] - centroid[:, None]) ** 2 * norm_mag).sum(axis=1))

        # Spectral rolloff 0.85
        cumsum = np.cumsum(norm_mag, axis=1)
        ro = (cumsum < 0.85).sum(axis=1).astype(np.float32)

        feats = np.stack([log_energy, zcr, centroid, bw, ro], axis=1)
        # Normalize roughly to a reasonable scale
        feats = feats.astype(np.float32)
        return feats


class VideoFeatureExtractor:
    """Compute 20-dim features per frame:
    - 16-bin grayscale histogram over detected face ROI (L1-normalized)
    - 4 geometry features: cx, cy, w, h normalized by frame size
    Fallback to center crop if face not detected.
    """
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def features_for_frame(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            roi = gray[y:y+fh, x:x+fw]
            cx = (x + fw / 2.0) / float(w)
            cy = (y + fh / 2.0) / float(h)
            nw = fw / float(w)
            nh = fh / float(h)
        else:
            # Fallback to center crop
            size = int(min(h, w) * 0.5)
            x = (w - size) // 2
            y = (h - size) // 2
            roi = gray[y:y+size, x:x+size]
            cx = 0.5; cy = 0.5; nw = size / float(w); nh = size / float(h)
        roi = cv2.resize(roi, (96, 96))
        hist = cv2.calcHist([roi], [0], None, [16], [0, 256]).flatten().astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        geom = np.array([cx, cy, nw, nh], dtype=np.float32)
        return np.concatenate([hist, geom], axis=0)  # (20,)


def make_sequence(arr_list, target_T):
    # arr_list: list of feature vectors [D]
    if not arr_list:
        return np.zeros((target_T, len(arr_list[0]) if arr_list else 1), dtype=np.float32)
    X = np.stack(arr_list, axis=0)
    # Simple resample by linear interpolation to target_T
    t_old = np.linspace(0, 1, num=X.shape[0], dtype=np.float32)
    t_new = np.linspace(0, 1, num=target_T, dtype=np.float32)
    # Interp each dim
    D = X.shape[1]
    Y = np.zeros((target_T, D), dtype=np.float32)
    for d in range(D):
        Y[:, d] = np.interp(t_new, t_old, X[:, d])
    return Y


def run_realtime(cfg_path, ckpt_path, device='cpu', camera_index=-1, backend='dshow'):
    device = torch.device(device)
    model, args = load_model(cfg_path, ckpt_path, device=device)

    # Expected input dims
    text_dim, video_dim, audio_dim = args['model']['input_dim']
    target_T = 50  # assume 1s window aligned_50

    cap = open_usb_camera(preferred_index=camera_index, backend=backend)
    if cap is None:
        print('ERROR: Cannot open webcam (USB). Try --camera N or --backend msmf/any')
        return
    afeat = AudioFeatureExtractor(sample_rate=16000, frame_ms=20, hop_ms=20)
    vfeat = VideoFeatureExtractor()

    audio_buf = RingBuffer(maxlen=16000)  # 1s of audio
    video_buf = RingBuffer(maxlen=target_T)  # keep last ~50 frames' feats

    # Audio stream
    if HAVE_SD:
        def audio_cb(indata, frames, time_info, status):
            if status:
                # print(status)
                pass
            mono = indata.copy().astype(np.float32)
            if mono.ndim > 1:
                mono = mono.mean(axis=1)
            for s in mono:
                audio_buf.append(float(s))
        stream = sd.InputStream(channels=1, samplerate=16000, callback=audio_cb, blocksize=0)
        stream.start()
    else:
        stream = None
        print('WARNING: sounddevice not available; using zero audio features')

    try:
        last_t = time.time()
        smoothing = deque(maxlen=5)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Collect video features
            vf = vfeat.features_for_frame(frame)
            video_buf.append(vf)

            # Every ~200 ms, run inference
            now = time.time()
            if now - last_t >= 0.2:
                last_t = now
                # Build 1s window sequences
                v_seq = make_sequence(video_buf.get_all(), target_T)  # [50, 20]
                if HAVE_SD:
                    a_samples = np.array(audio_buf.get_all(), dtype=np.float32)
                    a_feats = afeat.features(a_samples)  # [T_a, 5]
                    a_seq = make_sequence(list(a_feats), target_T)  # [50, 5]
                else:
                    a_seq = np.zeros((target_T, audio_dim), dtype=np.float32)

                # Text as zeros
                t_seq = np.zeros((target_T, text_dim), dtype=np.float32)

                # Make batch tensors
                t = torch.from_numpy(t_seq).unsqueeze(0).to(device)
                v = torch.from_numpy(v_seq).unsqueeze(0).to(device)
                a = torch.from_numpy(a_seq).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(t, v, a)
                    score = out['sentiment_preds'].squeeze().item()
                smoothing.append(score)
                disp_score = float(np.mean(smoothing))

                # Overlay on frame
                txt = f"Sentiment: {disp_score:+.2f}"
                color = (0, 255, 0) if disp_score >= 0 else (0, 0, 255)
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

            cv2.imshow('GFMamba Real-Time Sentiment', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
    finally:
        if HAVE_SD and stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='GFMamba Real-Time Sentiment (webcam+mic)')
    parser.add_argument('--config', default='configs/mosi_train.yaml')
    parser.add_argument('--ckpt', default='ckpt/mosi/best_valid_model_seed_42.pth')
    parser.add_argument('--device', default='cpu'); parser.add_argument('--camera', type=int, default=-1); parser.add_argument('--backend', choices=['dshow','msmf','any'], default='dshow'); parser.add_argument('--list-cameras', action='store_true', help='List available camera indices and exit')
    args = parser.parse_args()
    if args.list_cameras:
        avail = list_cameras(10, backend=args.backend)
        print(f'Available cameras (backend={args.backend}): {avail}')
        return
    run_realtime(args.config, args.ckpt, args.device, camera_index=args.camera, backend=args.backend)


if __name__ == '__main__':
    main()

