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
    for i in order:
        cap = open_camera_by_index(i, backend)
        if cap is not None:
            return cap
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


# --- Audio device helpers ---

def list_audio_devices():
    if not HAVE_SD:
        print('sounddevice not installed, cannot list audio devices')
        return
    try:
        devices = sd.query_devices()
        default = sd.default.device
        print('Audio devices:')
        for i, d in enumerate(devices):
            mark = ''
            if isinstance(default, (list, tuple)) and i == default[0]:
                mark = '(default in)'
            print(f'  [{i:02d}] {d.get("name", "?")}   in={d.get("max_input_channels", 0)} out={d.get("max_output_channels", 0)} {mark}')
    except Exception as e:
        print('Failed to query audio devices:', repr(e))


# --- Config/model loading ---

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


# --- Utility: running standardization ---
class RunningStandardizer:
    """Keeps running mean and std (per-dim) and applies z-score online.
    Uses Welford's algorithm; eps avoids division by zero.
    """
    def __init__(self, dim, eps=1e-6):
        self.dim = dim
        self.eps = eps
        self.count = 0
        self.mean = np.zeros((dim,), dtype=np.float64)
        self.M2 = np.zeros((dim,), dtype=np.float64)

    def update(self, x):
        # x: [T, D] or [D]
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        for row in arr:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.M2 += delta * delta2

    def std(self):
        if self.count < 2:
            return np.ones_like(self.mean)
        var = self.M2 / max(1, self.count - 1)
        std = np.sqrt(np.maximum(var, self.eps))
        return std

    def apply(self, x):
        arr = np.asarray(x, dtype=np.float32)
        if self.count < 2:
            return arr.astype(np.float32)
        return ((arr - self.mean.astype(np.float32)) / (self.std().astype(np.float32))).astype(np.float32)


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
        feats = feats.astype(np.float32)
        return feats


class VideoFeatureExtractor:
    """Compute 20-dim features per frame:
    - 16-bin grayscale histogram over detected face ROI (L1-normalized)
    - 4 geometry features: cx, cy, w, h normalized by frame size
    Fallback to center crop if face not detected.
    Returns (features, bbox, used_face).
    """
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def features_for_frame(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        used_face = False
        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            used_face = True
            roi = gray[y:y+fh, x:x+fw]
            cx = (x + fw / 2.0) / float(w)
            cy = (y + fh / 2.0) / float(h)
            nw = fw / float(w)
            nh = fh / float(h)
            bbox = (int(x), int(y), int(fw), int(fh))
        else:
            # Fallback to center crop
            size = int(min(h, w) * 0.5)
            x = (w - size) // 2
            y = (h - size) // 2
            roi = gray[y:y+size, x:x+size]
            cx = 0.5; cy = 0.5; nw = size / float(w); nh = size / float(h)
            bbox = (int(x), int(y), int(size), int(size))
        roi = cv2.resize(roi, (96, 96))
        hist = cv2.calcHist([roi], [0], None, [16], [0, 256]).flatten().astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        geom = np.array([cx, cy, nw, nh], dtype=np.float32)
        feat = np.concatenate([hist, geom], axis=0)  # (20,)
        return feat, bbox, used_face


# --- Sequence builder ---

def make_sequence(arr_list, target_T):
    # arr_list: list of feature vectors [D]
    if not arr_list:
        # Default to zeros of the expected dimensionality if we can infer it; else dim 1
        D = len(arr_list[0]) if arr_list else 1
        return np.zeros((target_T, D), dtype=np.float32)
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


# --- Main loop ---

def run_realtime(cfg_path, ckpt_path, device='cpu', camera_index=-1, backend='dshow', audio_device=None, disable_audio=False, disable_video=False, std_online=True):
    device = torch.device(device)
    model, args = load_model(cfg_path, ckpt_path, device=device)

    # Expected input dims
    text_dim, video_dim, audio_dim = args['model']['input_dim']
    target_T = 50  # assume 1s window aligned_50

    # Video
    cap = None
    vfeat = None
    if not disable_video:
        cap = open_usb_camera(preferred_index=camera_index, backend=backend)
        if cap is None:
            print('WARNING: Cannot open webcam (USB). Falling back to no-video mode. Try --camera N or --backend msmf/any')
            disable_video = True
        else:
            vfeat = VideoFeatureExtractor()

    # Audio
    afeat = AudioFeatureExtractor(sample_rate=16000, frame_ms=20, hop_ms=20)

    audio_buf = RingBuffer(maxlen=16000)  # 1s of audio
    video_buf = RingBuffer(maxlen=target_T)  # keep last ~50 frames' feats

    # Running standardizers (online z-score)
    a_std = RunningStandardizer(dim=audio_dim)
    v_std = RunningStandardizer(dim=video_dim)

    # Audio stream
    stream = None
    mic_ok = False
    if HAVE_SD and not disable_audio:
        try:
            stream = sd.InputStream(channels=1, samplerate=16000, dtype='float32', device=audio_device, callback=lambda indata, frames, time_info, status: _audio_cb(indata, status, audio_buf))
            stream.start()
            mic_ok = True
        except Exception as e:
            print('WARNING: Failed to open microphone stream:', repr(e))
            mic_ok = False
    else:
        if not HAVE_SD and not disable_audio:
            print('WARNING: sounddevice not available; using zero audio features')
        elif disable_audio:
            print('Audio disabled by flag; using zero audio features')

    try:
        last_t = time.time()
        smoothing = deque(maxlen=5)
        first_ts = time.time()
        # Allow an initial calibration period to collect stats
        CALIB_SECS = 1.5
        while True:
            frame = None
            if not disable_video and cap is not None:
                ret, frame = cap.read()
                if not ret:
                    frame = None
            if frame is not None:
                # Collect video features
                vf, bbox, used_face = vfeat.features_for_frame(frame)
            video_buf.append(vf)
            # Draw the bbox used for features (green if face-detected, yellow if fallback)
            if bbox is not None:
                x, y, bw, bh = bbox
                color = (0, 255, 0) if used_face else (0, 255, 255)
                try:
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                except Exception:
                    pass

            # Every ~200 ms, run inference
            now = time.time()
            if now - last_t >= 0.2:
                last_t = now
                # Build 1s window sequences
                if not disable_video and len(video_buf.get_all()) > 0:
                    v_seq = make_sequence(video_buf.get_all(), target_T)  # [50, 20]
                    # Update video std
                    if std_online:
                        v_std.update(v_seq)
                        v_seq = v_std.apply(v_seq)
                else:
                    v_seq = np.zeros((target_T, video_dim), dtype=np.float32)

                if mic_ok and len(audio_buf.get_all()) > 0 and not disable_audio:
                    a_samples = np.array(audio_buf.get_all(), dtype=np.float32)
                    a_feats = afeat.features(a_samples)  # [T_a, 5]
                    a_seq = make_sequence(list(a_feats), target_T)  # [50, 5]
                    # Update audio std
                    if std_online:
                        a_std.update(a_seq)
                        a_seq = a_std.apply(a_seq)
                else:
                    a_seq = np.zeros((target_T, audio_dim), dtype=np.float32)

                # Text as zeros (no ASR). Note: projector bias will inject a prior; that's fine.
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

                # Compose overlay text
                overlay_lines = [f"Sentiment: {disp_score:+.2f}"]
                # Audio level meter
                if not disable_audio:
                    rms = float(np.sqrt(np.mean(np.square(np.array(audio_buf.get_all(), dtype=np.float32)))) or 0.0)
                    db = 20.0 * np.log10(max(rms, 1e-6))
                    overlay_lines.append(f"Mic: {db:6.1f} dB {'OK' if db>-40 else 'LOW'}")
                if not disable_video:
                    overlay_lines.append(f"Frames: {len(video_buf.get_all()):2d}")

                if frame is None:
                    # If no frame, create a blank canvas to show text
                    frame = np.zeros((240, 320, 3), dtype=np.uint8)

                # Draw overlay
                y0 = 30
                for line in overlay_lines:
                    color = (0, 255, 0) if 'Sentiment' in line and disp_score >= 0 else (0, 0, 255) if 'Sentiment' in line else (255,255,255)
                    cv2.putText(frame, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                    y0 += 24

            # Show frame if we have one (or blank)
            if frame is not None:
                cv2.imshow('GFMamba Real-Time Sentiment', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
            else:
                # No video: still poll for quit
                if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                    break
    finally:
        if HAVE_SD and stream is not None:
            try:
                stream.stop(); stream.close()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def _audio_cb(indata, status, audio_buf: RingBuffer):
    if status:
        # print(status)
        pass
    mono = indata.astype(np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=1)
    for s in mono:
        audio_buf.append(float(s))


def main():
    parser = argparse.ArgumentParser(description='GFMamba Real-Time Sentiment (webcam+mic)')
    parser.add_argument('--config', default='configs/mosi_train.yaml')
    parser.add_argument('--ckpt', default='ckpt/mosi/best_valid_model_seed_42.pth')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--camera', type=int, default=-1)
    parser.add_argument('--backend', choices=['dshow','msmf','any'], default='dshow')
    parser.add_argument('--list-cameras', action='store_true', help='List available camera indices and exit')
    parser.add_argument('--list-audio', action='store_true', help='List audio input devices and exit')
    parser.add_argument('--audio-device', type=int, default=None, help='Audio input device index (see --list-audio)')
    parser.add_argument('--no-audio', action='store_true', help='Disable microphone/audio features')
    parser.add_argument('--no-video', action='store_true', help='Disable webcam/video features')
    parser.add_argument('--no-std', action='store_true', help='Disable online standardization (z-score) of features')
    args = parser.parse_args()

    if args.list_cameras:
        avail = list_cameras(10, backend=args.backend)
        print(f'Available cameras (backend={args.backend}): {avail}')
        return
    if args.list_audio:
        list_audio_devices();
        return
    run_realtime(args.config, args.ckpt, args.device, camera_index=args.camera, backend=args.backend,
                 audio_device=args.audio_device, disable_audio=args.no_audio, disable_video=args.no_video,
                 std_online=(not args.no_std))


if __name__ == '__main__':
    main()
