#!/usr/bin/env python3
# stabilized_rppg_v5_1_facemesh_hrv.py
#
# rPPG v5.1:
#   - FaceMesh ROI (forehead + cheeks polygons, eye/mouth removed) + skin mask
#   - POS + CHROM overlap-add ensemble
#   - True beat stream RRIs + artifact correction
#   - NN filtering + interpolation
#   - "Best window" selector for 30s & 60s (confidence/motion/kept%)
#   - Frequency-domain HRV (LF/HF/LFHF) via Welch on interpolated tachogram
#
# Outputs:
#   rppg_hr_bpm.csv
#   rppg_rr_intervals_ms.csv
#   rppg_nn_intervals_ms.csv
#   rppg_nn_intervals_interpolated_ms.csv
#   rppg_hrv_summary.csv
#   rppg_results.npz
#   rppg_debug.mp4
#
# Headless: no cv2.imshow. Close matplotlib plot to stop early.

import kagglehub
import cv2
import numpy as np
import os
import glob
import csv
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import butter, filtfilt, detrend, find_peaks, welch

import mediapipe as mp


# -----------------------------
# Signal helpers
# -----------------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def safe_norm(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return (x - np.mean(x)) / (np.std(x) + eps)

def hann(n):
    return np.hanning(n).astype(np.float32)

def parabolic_interpolation(mag, idx):
    if idx <= 0 or idx >= len(mag) - 1:
        return float(idx)
    y0, y1, y2 = mag[idx - 1], mag[idx], mag[idx + 1]
    denom = (y0 - 2*y1 + y2)
    if abs(denom) < 1e-12:
        return float(idx)
    delta = 0.5 * (y0 - y2) / denom
    return float(idx) + float(delta)

def compute_bpm_from_fft(sig, fs, bpm_min=45, bpm_max=200, prior_bpm=None, prior_window_bpm=15):
    n = len(sig)
    x = sig * hann(n)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X).astype(np.float32)

    fmin = bpm_min / 60.0
    fmax = bpm_max / 60.0
    band = (freqs >= fmin) & (freqs <= fmax)
    mag[~band] = 0.0

    if prior_bpm is not None and prior_bpm > 40:
        last_f = prior_bpm / 60.0
        w = prior_window_bpm / 60.0
        dist = np.abs(freqs - last_f)
        penalty = np.clip(dist / w, 0.0, 1.0)
        mag = mag * (1.0 - 0.6 * penalty)

    peak_idx = int(np.argmax(mag))
    peak_idx_f = parabolic_interpolation(mag, peak_idx)
    peak_freq = np.interp(peak_idx_f, np.arange(len(freqs), dtype=np.float32), freqs)
    bpm = float(peak_freq * 60.0)

    band_vals = mag[band]
    med = float(np.median(band_vals) + 1e-8)
    peak = float(mag[peak_idx] + 1e-8)
    conf = peak / med

    return bpm, conf


# -----------------------------
# rPPG overlap-add waveforms
# -----------------------------
def pos_overlap_add(rgb_means, fs, window_sec=1.6):
    X = np.asarray(rgb_means, dtype=np.float32)
    T = X.shape[0]
    L = max(32, int(window_sec * fs))
    if T < L:
        return None

    H = np.zeros(T, dtype=np.float32)
    W = np.zeros(T, dtype=np.float32)
    win = hann(L)

    for t0 in range(0, T - L + 1):
        C = X[t0:t0+L, :]
        Cn = C / (np.mean(C, axis=0, keepdims=True) + 1e-8)

        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = -2 * Cn[:, 0] + Cn[:, 1] + Cn[:, 2]

        alpha = (np.std(S1) + 1e-8) / (np.std(S2) + 1e-8)
        h = S1 - alpha * S2
        h = safe_norm(h)

        H[t0:t0+L] += h * win
        W[t0:t0+L] += win

    return H / (W + 1e-8)

def chrom_overlap_add(rgb_means, fs, window_sec=1.6):
    X = np.asarray(rgb_means, dtype=np.float32)
    T = X.shape[0]
    L = max(32, int(window_sec * fs))
    if T < L:
        return None

    H = np.zeros(T, dtype=np.float32)
    W = np.zeros(T, dtype=np.float32)
    win = hann(L)

    for t0 in range(0, T - L + 1):
        C = X[t0:t0+L, :]
        Cn = C / (np.mean(C, axis=0, keepdims=True) + 1e-8)

        R, G, B = Cn[:, 0], Cn[:, 1], Cn[:, 2]
        Xc = 3*R - 2*G
        Yc = 1.5*R + G - 1.5*B

        alpha = (np.std(Xc) + 1e-8) / (np.std(Yc) + 1e-8)
        h = Xc - alpha * Yc
        h = safe_norm(h)

        H[t0:t0+L] += h * win
        W[t0:t0+L] += win

    return H / (W + 1e-8)


# -----------------------------
# Skin mask (optional extra cleanup)
# -----------------------------
def skin_mask_ycrcb(bgr):
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 130, 70], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


# -----------------------------
# HRV utilities
# -----------------------------
def hrv_metrics(rr_ms):
    rr = np.array(rr_ms, dtype=np.float32)
    if rr.size < 3:
        return {"rmssd": np.nan, "sdnn": np.nan, "pnn50": np.nan, "n": int(rr.size)}
    diff = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff * diff)))
    sdnn = float(np.std(rr, ddof=1)) if rr.size >= 2 else float(np.nan)
    pnn50 = float(np.mean(np.abs(diff) > 50.0) * 100.0)
    return {"rmssd": rmssd, "sdnn": sdnn, "pnn50": pnn50, "n": int(rr.size)}

def nn_filter_with_mask(rr_list_ms, pct=0.25):
    rr = np.array(rr_list_ms, dtype=np.float32)
    n = int(rr.size)
    if n == 0:
        return [], [], 0.0, np.nan
    if n < 5:
        keep = [(300.0 <= float(x) <= 2000.0) for x in rr]
        nn = [float(rr[i]) for i in range(n) if keep[i]]
        kept_pct = 100.0 * (sum(keep) / max(1, n))
        med = float(np.median(rr))
        return nn, keep, kept_pct, med

    med = float(np.median(rr))
    keep = []
    for x in rr:
        xf = float(x)
        ok = (300.0 <= xf <= 2000.0) and (abs(xf - med) <= pct * med)
        keep.append(bool(ok))

    nn = [float(rr[i]) for i in range(n) if keep[i]]
    kept_pct = 100.0 * (sum(keep) / max(1, n))
    return nn, keep, kept_pct, med

def interpolate_gaps(times, values, keep_mask):
    t = np.asarray(times, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)
    keep = np.asarray(keep_mask, dtype=bool)

    if t.size == 0 or keep.size != t.size:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    idx_keep = np.where(keep)[0]
    if idx_keep.size < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    i0 = int(idx_keep[0])
    i1 = int(idx_keep[-1])

    t_seg = t[i0:i1+1]
    v_seg = v[i0:i1+1]
    keep_seg = keep[i0:i1+1]

    knots_x = t_seg[keep_seg]
    knots_y = v_seg[keep_seg]
    v_interp = np.interp(t_seg, knots_x, knots_y).astype(np.float32)

    return t_seg.astype(np.float32), v_interp

def lf_hf_from_interpolated_tachogram(t_sec, nn_ms, fs_resample=4.0):
    """
    Compute LF/HF using Welch PSD on a uniformly resampled tachogram.
    Input: t_sec (seconds), nn_ms (ms) sampled irregularly but interpolated already.
    We still resample to uniform fs_resample grid for clean PSD.
    """
    if t_sec is None or nn_ms is None:
        return {"lf": np.nan, "hf": np.nan, "lfhf": np.nan}

    t = np.asarray(t_sec, dtype=np.float32)
    y = np.asarray(nn_ms, dtype=np.float32)
    if t.size < 10:
        return {"lf": np.nan, "hf": np.nan, "lfhf": np.nan}

    t0, t1 = float(t[0]), float(t[-1])
    if (t1 - t0) < 20.0:  # too short for meaningful LF
        return {"lf": np.nan, "hf": np.nan, "lfhf": np.nan}

    tg = np.arange(t0, t1, 1.0 / fs_resample, dtype=np.float32)
    if tg.size < 32:
        return {"lf": np.nan, "hf": np.nan, "lfhf": np.nan}

    yg = np.interp(tg, t, y).astype(np.float32)
    yg = yg - np.mean(yg)

    nperseg = int(min(256, max(64, (len(yg) // 2))))
    f, pxx = welch(yg, fs=fs_resample, nperseg=nperseg, noverlap=nperseg // 2)

    def band_power(f, pxx, f0, f1):
        m = (f >= f0) & (f < f1)
        if not np.any(m):
            return 0.0
        return float(np.trapz(pxx[m], f[m]))

    lf = band_power(f, pxx, 0.04, 0.15)
    hf = band_power(f, pxx, 0.15, 0.40)
    lfhf = float(lf / (hf + 1e-12)) if hf > 0 else np.nan

    return {"lf": lf, "hf": hf, "lfhf": lfhf}


# -----------------------------
# FaceMesh ROI builder
# -----------------------------
class FaceMeshROI:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.fm = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.prev_xy = None
        self.motion_ema = 0.0
        self.motion_alpha = 0.15

        self.left_cheek = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]
        self.right_cheek = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58]
        self.forehead = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                         400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103]

    def _landmarks_to_xy(self, landmarks, w, h):
        xy = np.zeros((len(landmarks.landmark), 2), dtype=np.float32)
        for i, lm in enumerate(landmarks.landmark):
            xy[i, 0] = lm.x * w
            xy[i, 1] = lm.y * h
        return xy

    def _poly(self, xy, idxs):
        pts = xy[idxs].astype(np.int32)
        return pts.reshape(-1, 1, 2)

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.fm.process(rgb)

        if not res.multi_face_landmarks:
            self.prev_xy = None
            return None, 0.0, None

        lms = res.multi_face_landmarks[0]
        xy = self._landmarks_to_xy(lms, w, h)

        if self.prev_xy is not None and self.prev_xy.shape == xy.shape:
            disp = np.linalg.norm(xy - self.prev_xy, axis=1)
            motion = float(np.median(disp))
            self.motion_ema = self.motion_alpha * motion + (1 - self.motion_alpha) * self.motion_ema
        self.prev_xy = xy

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, self._poly(xy, self.left_cheek), 255)
        cv2.fillConvexPoly(mask, self._poly(xy, self.right_cheek), 255)
        cv2.fillConvexPoly(mask, self._poly(xy, self.forehead), 255)

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [263, 387, 385, 362, 380, 373]
        mouth = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        cv2.fillConvexPoly(mask, self._poly(xy, left_eye), 0)
        cv2.fillConvexPoly(mask, self._poly(xy, right_eye), 0)
        cv2.fillConvexPoly(mask, self._poly(xy, mouth), 0)

        skin = skin_mask_ycrcb(frame_bgr)
        mask = cv2.bitwise_and(mask, skin)

        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        ys, xs = np.where(mask > 0)
        if len(xs) > 50:
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            debug = (x0, y0, x1 - x0, y1 - y0)
        else:
            debug = None

        return mask, float(self.motion_ema), debug


# -----------------------------
# Main pipeline
# -----------------------------
class RPPG_v51_FaceMeshHRV:
    def __init__(self, video_path, gt_path):
        print("--- INITIALIZING rPPG v5.1 (FaceMesh ROI + HRV + best windows + LF/HF, headless) ---")
        self.cap = cv2.VideoCapture(video_path)
        self.roi = FaceMeshROI()

        with open(gt_path, "r") as f:
            lines = f.readlines()
        gt_vals = [float(x) for x in lines[1].strip().split()]
        self.gt = np.array(gt_vals, dtype=np.float32)

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        print(f"[gt] loaded {len(self.gt)} values from ground_truth.txt")
        print(f"[video] fps={self.fps:.2f}")
        print(f"[video] frames={self.frame_count} duration_sec={self.frame_count/self.fps:.1f} gt_len={len(self.gt)}")
        print("[gt] min/max/mean/std:",
              float(self.gt.min()), float(self.gt.max()), float(self.gt.mean()), float(self.gt.std()))

        # HR estimation window
        self.window_sec = 20.0
        self.buf_len = int(self.window_sec * self.fps)
        self.rgb_buf = deque(maxlen=self.buf_len)

        # Beat-stream pulse buffer
        self.pulse_buf_sec = 15.0
        self.pulse_len = int(self.pulse_buf_sec * self.fps)
        self.pulse_stream_raw = deque(maxlen=self.pulse_len)
        self.last_peak_global_idx = None

        # Streams
        self.rr_stream = []   # (t_sec_peak, rr_ms)
        self.rows_rri = []
        self.rr_recent = deque(maxlen=7)

        self.rows_hr = []     # time series rows for rppg_hr_bpm.csv

        # tracking vars
        self.idx = -1
        self.current_bpm = 0.0
        self.confidence = 0.0
        self.last_method = "NA"
        self.errors = []

        # plotting
        self.plot_ai = deque(maxlen=250)
        self.plot_gt = deque(maxlen=250)

        self.last_h30 = None
        self.last_h60 = None

        # store per-sample for best-window scoring
        self.hr_time = []      # seconds
        self.hr_conf = []      # confidence
        self.hr_motion = []    # motion_ema
        self.hr_kept60 = []    # kept% last60 (to guide selection; not required)
        self.hr_kept30 = []    # kept% last30

    def gt_window_avg(self):
        if self.idx < 0:
            return None
        start = max(0, self.idx - self.buf_len + 1)
        return float(np.mean(self.gt[start:self.idx + 1]))

    def _build_raw_waveforms(self):
        if len(self.rgb_buf) < int(6.0 * self.fps):
            return None, None
        X = np.array(self.rgb_buf, dtype=np.float32)
        pos_raw = pos_overlap_add(X, fs=self.fps, window_sec=1.6)
        chr_raw = chrom_overlap_add(X, fs=self.fps, window_sec=1.6)
        if pos_raw is None or chr_raw is None:
            return None, None
        return detrend(pos_raw).astype(np.float32), detrend(chr_raw).astype(np.float32)

    def estimate_hr_and_method(self):
        pos_raw, chr_raw = self._build_raw_waveforms()
        if pos_raw is None:
            return None, 0.0, "NA", None

        try:
            sp = safe_norm(butter_bandpass_filter(pos_raw, 0.75, 3.5, self.fps, order=3))
            sc = safe_norm(butter_bandpass_filter(chr_raw, 0.75, 3.5, self.fps, order=3))
        except Exception:
            return None, 0.0, "NA", None

        prior = self.current_bpm if self.current_bpm > 40 else None
        bpm_p, conf_p = compute_bpm_from_fft(sp, self.fps, prior_bpm=prior)
        bpm_c, conf_c = compute_bpm_from_fft(sc, self.fps, prior_bpm=prior)

        if conf_c > conf_p:
            return bpm_c, conf_c, "CHROM", chr_raw
        return bpm_p, conf_p, "POS", pos_raw

    def _append_pulse_sample(self, chosen_raw):
        if chosen_raw is None or len(chosen_raw) == 0:
            return
        self.pulse_stream_raw.append(float(chosen_raw[-1]))

    def _emit_rr_corrected(self, t_sec, rr_ms):
        if rr_ms < 300.0 or rr_ms > 2000.0:
            return

        if len(self.rr_recent) >= 4:
            med = float(np.median(self.rr_recent))
            if med > 1e-6:
                if rr_ms < 0.70 * med:
                    return
                if 1.70 * med <= rr_ms <= 2.50 * med:
                    half = rr_ms / 2.0
                    for _ in range(2):
                        self.rows_rri.append([t_sec, half])
                        self.rr_stream.append((t_sec, half))
                        self.rr_recent.append(half)
                    return
                if rr_ms > 1.60 * med:
                    return

        self.rows_rri.append([t_sec, rr_ms])
        self.rr_stream.append((t_sec, rr_ms))
        self.rr_recent.append(rr_ms)

    def _detect_new_peaks_and_emit_rr(self):
        if len(self.pulse_stream_raw) < int(8.0 * self.fps):
            return

        raw = detrend(np.array(self.pulse_stream_raw, dtype=np.float32))
        try:
            filt = safe_norm(butter_bandpass_filter(raw, 0.75, 3.5, self.fps, order=3))
        except Exception:
            return

        L = len(filt)
        edge = int(0.35 * self.fps)
        if L <= 2 * edge + 10:
            return

        core = filt[edge:L - edge]
        core_offset = edge

        if self.current_bpm > 40:
            max_bpm = float(np.clip(self.current_bpm * 1.35, 110.0, 170.0))
        else:
            max_bpm = 170.0
        min_dist = max(1, int(self.fps * 60.0 / max_bpm))

        peaks, _ = find_peaks(core, distance=min_dist, prominence=0.35)
        if peaks is None or len(peaks) < 2:
            return

        buf_peaks = [int(p) + core_offset for p in peaks]
        global_peaks = [self.idx - (L - 1 - bp) for bp in buf_peaks]

        if self.last_peak_global_idx is None:
            self.last_peak_global_idx = global_peaks[-1]
            return

        new_peaks = [gp for gp in global_peaks if gp > self.last_peak_global_idx]
        if not new_peaks:
            return

        prev = self.last_peak_global_idx
        for gp in new_peaks:
            rr_ms = float(((gp - prev) / self.fps) * 1000.0)
            t_sec = gp / self.fps
            self._emit_rr_corrected(t_sec, rr_ms)
            prev = gp

        self.last_peak_global_idx = new_peaks[-1]

    def _rr_in_window(self, t0, t1):
        return [(t, rr) for (t, rr) in self.rr_stream if (t >= t0 and t <= t1)]

    def _hrv_for_window(self, t0, t1, nn_pct=0.25):
        rr_pairs = self._rr_in_window(t0, t1)
        if not rr_pairs:
            return None

        times = [float(t) for (t, _) in rr_pairs]
        rr = [float(x) for (_, x) in rr_pairs]
        nn, keep_mask, kept_pct, _ = nn_filter_with_mask(rr, pct=nn_pct)
        m = hrv_metrics(nn)
        m["kept_pct"] = float(kept_pct)
        m["n_raw"] = int(len(rr))

        # LF/HF from interpolated tachogram of NN-only (gap-filled)
        t_interp, rr_interp = interpolate_gaps(times, rr, keep_mask)
        spec = lf_hf_from_interpolated_tachogram(t_interp, rr_interp, fs_resample=4.0)
        m["lf"] = spec["lf"]
        m["hf"] = spec["hf"]
        m["lfhf"] = spec["lfhf"]

        return m

    def _mean_conf_motion_in_window(self, t0, t1):
        # Uses hr_time arrays (recorded only when HR is valid)
        if len(self.hr_time) == 0:
            return np.nan, np.nan
        t = np.array(self.hr_time, dtype=np.float32)
        m = (t >= t0) & (t <= t1)
        if not np.any(m):
            return np.nan, np.nan
        conf = float(np.mean(np.array(self.hr_conf, dtype=np.float32)[m]))
        motion = float(np.mean(np.array(self.hr_motion, dtype=np.float32)[m]))
        return conf, motion

    def _score_window(self, t0, t1, hrv_m):
        # score prefers: high conf, low motion, high kept%, and enough beats
        if hrv_m is None:
            return -1e9
        conf, motion = self._mean_conf_motion_in_window(t0, t1)
        if not (conf == conf) or not (motion == motion):
            return -1e9
        kept = float(hrv_m.get("kept_pct", 0.0))
        n = int(hrv_m.get("n", 0))
        # normalize terms
        conf_term = np.clip(conf / 12.0, 0.0, 2.0)
        motion_term = 1.0 / (1.0 + (motion / 2.0))  # motion in px; smaller is better
        kept_term = np.clip(kept / 100.0, 0.0, 1.0)
        n_term = np.clip(n / 40.0, 0.0, 1.0)       # "enough beats" bonus
        return float(conf_term * motion_term * kept_term * (0.6 + 0.4 * n_term))

    def find_best_window(self, win_sec=60.0, step_sec=1.0, nn_pct=0.25):
        # Search only within the span where we have HR timestamps
        if len(self.hr_time) < 10:
            return None

        t_start = float(self.hr_time[0])
        t_end = float(self.hr_time[-1])

        if (t_end - t_start) < (win_sec + 5.0):
            return None

        best = {"score": -1e9, "t0": None, "t1": None, "m": None, "conf": None, "motion": None}
        t0 = t_start
        while (t0 + win_sec) <= t_end:
            t1 = t0 + win_sec
            m = self._hrv_for_window(t0, t1, nn_pct=nn_pct)
            if m is not None and m["n"] >= max(10, int(win_sec * (self.current_bpm / 60.0) * 0.4)):
                s = self._score_window(t0, t1, m)
                if s > best["score"]:
                    conf, motion = self._mean_conf_motion_in_window(t0, t1)
                    best.update({"score": s, "t0": t0, "t1": t1, "m": m, "conf": conf, "motion": motion})
            t0 += step_sec

        if best["t0"] is None:
            return None
        return best

    def run(self):
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 3))
        line_ai, = ax.plot([], [], label="AI HR (ensemble)")
        line_gt, = ax.plot([], [], label="GT HR (window-avg)")
        ax.set_ylim(40, 140)
        ax.set_xlim(0, 250)
        ax.legend()
        plt.title("rPPG v5.1: FaceMesh ROI + Best HRV Windows + LF/HF (Headless)")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = "rppg_debug.mp4"
        fw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(out_path, fourcc, self.fps, (fw, fh))
        print(f"[video] writing annotated output to {out_path}")
        print("-> Running (no cv2.imshow). Close the plot window to stop.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.idx += 1
            if self.idx >= len(self.gt):
                break

            mask, motion_ema, debug_bbox = self.roi.process(frame)

            if mask is not None:
                pix = frame[mask > 0]
                if pix.shape[0] > 200:
                    mean_bgr = np.mean(pix, axis=0)
                    mean_rgb = mean_bgr[::-1]
                    self.rgb_buf.append(mean_rgb)

                if debug_bbox is not None:
                    x, y, w, h = debug_bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                overlay = frame.copy()
                overlay[mask > 0] = (overlay[mask > 0] * 0.75 + np.array([0, 50, 0], dtype=np.float32)).astype(np.uint8)
                frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

            bpm, conf, method, chosen_raw = self.estimate_hr_and_method()

            # Confidence gate
            if bpm is not None and conf < 3.0:
                bpm = None
                chosen_raw = None

            if bpm is not None:
                motion_factor = float(np.clip(1.0 - (motion_ema / 6.0), 0.15, 1.0))
                conf_clamped = float(np.clip(conf / 10.0, 0.0, 1.0))
                alpha = (0.04 + 0.30 * conf_clamped) * motion_factor

                if self.current_bpm == 0.0:
                    self.current_bpm = float(bpm)
                else:
                    delta = bpm - self.current_bpm
                    max_jump = 8.0
                    if abs(delta) > max_jump:
                        bpm = self.current_bpm + max_jump * (1 if delta > 0 else -1)
                    self.current_bpm = alpha * float(bpm) + (1 - alpha) * self.current_bpm

                self.confidence = float(conf)
                self.last_method = method

                self._append_pulse_sample(chosen_raw)
                if (self.idx % max(1, int(self.fps // 10))) == 0:
                    self._detect_new_peaks_and_emit_rr()

            gt_win = self.gt_window_avg()
            if self.current_bpm > 30 and gt_win is not None:
                self.errors.append(abs(self.current_bpm - gt_win))

                self.plot_ai.append(self.current_bpm)
                self.plot_gt.append(gt_win)
                line_ai.set_data(range(len(self.plot_ai)), self.plot_ai)
                line_gt.set_data(range(len(self.plot_gt)), self.plot_gt)
                fig.canvas.draw()
                fig.canvas.flush_events()

                # "last" HRV windows
                t_now = self.idx / self.fps
                h30 = self._hrv_for_window(max(0.0, t_now - 30.0), t_now, nn_pct=0.25)
                h60 = self._hrv_for_window(max(0.0, t_now - 60.0), t_now, nn_pct=0.25)
                self.last_h30 = h30
                self.last_h60 = h60

                def v_or_blank(x):
                    return "" if x is None or not (x == x) else float(x)

                # record "best-window scoring traces"
                self.hr_time.append(t_now)
                self.hr_conf.append(float(self.confidence))
                self.hr_motion.append(float(motion_ema))
                self.hr_kept30.append(float(h30["kept_pct"]) if h30 else 0.0)
                self.hr_kept60.append(float(h60["kept_pct"]) if h60 else 0.0)

                # write row
                self.rows_hr.append([
                    t_now,
                    float(self.current_bpm),
                    float(self.confidence),
                    self.last_method,
                    float(gt_win),
                    float(motion_ema),

                    v_or_blank(h30["rmssd"]) if h30 else "",
                    v_or_blank(h30["sdnn"]) if h30 else "",
                    v_or_blank(h30["pnn50"]) if h30 else "",
                    int(h30["n"]) if h30 else 0,
                    float(h30["kept_pct"]) if h30 else 0.0,

                    v_or_blank(h60["rmssd"]) if h60 else "",
                    v_or_blank(h60["sdnn"]) if h60 else "",
                    v_or_blank(h60["pnn50"]) if h60 else "",
                    int(h60["n"]) if h60 else 0,
                    float(h60["kept_pct"]) if h60 else 0.0,
                ])

                cv2.putText(frame, f"HR: {self.current_bpm:.1f} ({self.last_method}) conf:{self.confidence:.2f}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"GT(win): {gt_win:.1f} motion:{motion_ema:.2f}px",
                            (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            writer.write(frame)

            if not plt.fignum_exists(fig.number):
                break

        self.cap.release()
        writer.release()
        plt.close()

        if self.errors:
            print(f"\nFINAL MAE (window-aligned): {float(np.mean(self.errors)):.2f} BPM")
        else:
            print("\nNo errors computed.")

        # Find best windows
        best30 = self.find_best_window(win_sec=30.0, step_sec=1.0, nn_pct=0.25)
        best60 = self.find_best_window(win_sec=60.0, step_sec=1.0, nn_pct=0.25)

        # Print last + best summaries
        def print_block(name, m, conf=None, motion=None, t0=None, t1=None):
            if m is None:
                print(f"[{name}] unavailable")
                return
            extra = ""
            if t0 is not None and t1 is not None:
                extra += f"  window=[{t0:.1f}s..{t1:.1f}s]"
            if conf is not None and motion is not None and (conf == conf) and (motion == motion):
                extra += f"  mean_conf={conf:.2f}  mean_motion={motion:.2f}px"
            print(f"[{name}] RMSSD={m['rmssd']:.2f}ms  SDNN={m['sdnn']:.2f}ms  pNN50={m['pnn50']:.1f}%  "
                  f"n={m['n']}  kept={m['kept_pct']:.1f}%  LF={m['lf']:.4g}  HF={m['hf']:.4g}  LF/HF={m['lfhf']:.3g}{extra}")

        # last windows:
        if self.last_h30:
            print_block("HRV last_30s (NN)", self.last_h30, None, None, None, None)
        if self.last_h60:
            print_block("HRV last_60s (NN)", self.last_h60, None, None, None, None)

        if best30:
            print_block("HRV best_30s (NN)", best30["m"], best30["conf"], best30["motion"], best30["t0"], best30["t1"])
        if best60:
            print_block("HRV best_60s (NN)", best60["m"], best60["conf"], best60["motion"], best60["t0"], best60["t1"])

        self.export(best30, best60, out_prefix="rppg")

    def export(self, best30, best60, out_prefix="rppg"):
        hr_csv = f"{out_prefix}_hr_bpm.csv"
        rri_csv = f"{out_prefix}_rr_intervals_ms.csv"
        nn_csv = f"{out_prefix}_nn_intervals_ms.csv"
        nn_interp_csv = f"{out_prefix}_nn_intervals_interpolated_ms.csv"
        summary_csv = f"{out_prefix}_hrv_summary.csv"
        npz_path = f"{out_prefix}_results.npz"

        # HR time series
        with open(hr_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "t_sec",
                "hr_bpm",
                "confidence",
                "method",
                "gt_bpm_windowavg",
                "motion_px_ema",
                "hrv30_rmssd_ms", "hrv30_sdnn_ms", "hrv30_pnn50", "hrv30_n", "hrv30_kept_pct",
                "hrv60_rmssd_ms", "hrv60_sdnn_ms", "hrv60_pnn50", "hrv60_n", "hrv60_kept_pct",
            ])
            w.writerows(self.rows_hr)

        # raw RRIs
        with open(rri_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_sec_peak", "rr_interval_ms"])
            w.writerows(self.rows_rri)

        # NN + interpolated NN (global)
        kept_pct_all = 0.0
        if self.rr_stream:
            t_all = [float(t) for (t, _) in self.rr_stream]
            rr_all = [float(rr) for (_, rr) in self.rr_stream]
            nn_all, keep_mask_all, kept_pct_all, _ = nn_filter_with_mask(rr_all, pct=0.25)

            with open(nn_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t_sec_peak", "nn_interval_ms"])
                for i, keep in enumerate(keep_mask_all):
                    if keep:
                        w.writerow([t_all[i], rr_all[i]])

            t_interp, rr_interp = interpolate_gaps(t_all, rr_all, keep_mask_all)
            with open(nn_interp_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t_sec_peak", "nn_interval_ms_interpolated"])
                for ti, ri in zip(t_interp.tolist(), rr_interp.tolist()):
                    w.writerow([float(ti), float(ri)])
        else:
            with open(nn_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["t_sec_peak", "nn_interval_ms"])
            with open(nn_interp_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f); w.writerow(["t_sec_peak", "nn_interval_ms_interpolated"])

        # Summary CSV: last + best windows
        def row_for(name, m, mean_conf=np.nan, mean_motion=np.nan, t0=np.nan, t1=np.nan):
            if m is None:
                return [name, "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
            return [
                name,
                float(t0) if (t0 == t0) else "",
                float(t1) if (t1 == t1) else "",
                float(mean_conf) if (mean_conf == mean_conf) else "",
                float(mean_motion) if (mean_motion == mean_motion) else "",
                float(m["kept_pct"]),
                int(m["n"]),
                float(m["rmssd"]),
                float(m["sdnn"]),
                float(m["pnn50"]),
                float(m["lf"]),
                float(m["hf"]),
                float(m["lfhf"]) if (m["lfhf"] == m["lfhf"]) else "",
            ]

        # Compute mean conf/motion for last windows using recorded HR arrays
        t_end = float(self.hr_time[-1]) if self.hr_time else np.nan
        last30_conf, last30_motion = self._mean_conf_motion_in_window(max(0.0, t_end - 30.0), t_end) if (t_end == t_end) else (np.nan, np.nan)
        last60_conf, last60_motion = self._mean_conf_motion_in_window(max(0.0, t_end - 60.0), t_end) if (t_end == t_end) else (np.nan, np.nan)

        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "segment",
                "t0_sec",
                "t1_sec",
                "mean_conf",
                "mean_motion_px",
                "nn_kept_pct",
                "n_nn",
                "rmssd_ms",
                "sdnn_ms",
                "pnn50_pct",
                "lf_power",
                "hf_power",
                "lf_hf_ratio",
            ])
            w.writerow(row_for("last_30s", self.last_h30, last30_conf, last30_motion, max(0.0, t_end - 30.0), t_end))
            w.writerow(row_for("last_60s", self.last_h60, last60_conf, last60_motion, max(0.0, t_end - 60.0), t_end))
            if best30:
                w.writerow(row_for("best_30s", best30["m"], best30["conf"], best30["motion"], best30["t0"], best30["t1"]))
            if best60:
                w.writerow(row_for("best_60s", best60["m"], best60["conf"], best60["motion"], best60["t0"], best60["t1"]))

        # NPZ export (same idea as before, plus best window info)
        hr_arr = np.array([r[1] for r in self.rows_hr], dtype=np.float32) if self.rows_hr else np.array([], dtype=np.float32)
        conf_arr = np.array([r[2] for r in self.rows_hr], dtype=np.float32) if self.rows_hr else np.array([], dtype=np.float32)
        motion_arr = np.array([r[5] for r in self.rows_hr], dtype=np.float32) if self.rows_hr else np.array([], dtype=np.float32)
        gt_arr = np.array([r[4] for r in self.rows_hr], dtype=np.float32) if self.rows_hr else np.array([], dtype=np.float32)
        t_hr = np.array([r[0] for r in self.rows_hr], dtype=np.float32) if self.rows_hr else np.array([], dtype=np.float32)

        if self.rr_stream:
            t_all = np.array([t for (t, _) in self.rr_stream], dtype=np.float32)
            rr_all = np.array([rr for (_, rr) in self.rr_stream], dtype=np.float32)
            nn_all, keep_mask_all, kept_pct_all, _ = nn_filter_with_mask(rr_all.tolist(), pct=0.25)
            keep_mask_all = np.array(keep_mask_all, dtype=np.uint8)
            nn_only = np.array(nn_all, dtype=np.float32)
            t_interp, rr_interp = interpolate_gaps(t_all.tolist(), rr_all.tolist(), keep_mask_all.tolist())
        else:
            t_all = np.array([], dtype=np.float32)
            rr_all = np.array([], dtype=np.float32)
            keep_mask_all = np.array([], dtype=np.uint8)
            nn_only = np.array([], dtype=np.float32)
            t_interp = np.array([], dtype=np.float32)
            rr_interp = np.array([], dtype=np.float32)
            kept_pct_all = 0.0

        def best_pack(best):
            if not best:
                return np.array([np.nan, np.nan], dtype=np.float32)
            return np.array([float(best["t0"]), float(best["t1"])], dtype=np.float32)

        np.savez(
            npz_path,
            fps=np.array([self.fps], dtype=np.float32),
            t_hr=t_hr,
            hr=hr_arr,
            hr_conf=conf_arr,
            motion=motion_arr,
            gt=gt_arr,
            rri_t=t_all,
            rri_ms=rr_all,
            nn_keep_mask=keep_mask_all,
            nn_only_ms=nn_only,
            nn_interp_t=np.asarray(t_interp, dtype=np.float32),
            nn_interp_ms=np.asarray(rr_interp, dtype=np.float32),
            nn_kept_pct_all=np.array([float(kept_pct_all)], dtype=np.float32),
            best30_t01=best_pack(best30),
            best60_t01=best_pack(best60),
        )

        print(f"[export] wrote {hr_csv}")
        print(f"[export] wrote {rri_csv}")
        print(f"[export] wrote {nn_csv}")
        print(f"[export] wrote {nn_interp_csv}")
        print(f"[export] wrote {summary_csv}")
        print(f"[export] wrote {npz_path}")


def get_ubfc_files():
    path = kagglehub.dataset_download("ashfakyeafi/ubfc-2")
    videos = glob.glob(os.path.join(path, "**", "*.avi"), recursive=True)
    for v in videos:
        d = os.path.dirname(v)
        gt = os.path.join(d, "ground_truth.txt")
        if os.path.exists(gt):
            return v, gt
    return None, None


if __name__ == "__main__":
    vid, gt = get_ubfc_files()
    if vid:
        RPPG_v51_FaceMeshHRV(vid, gt).run()
    else:
        print("Data not found.")
