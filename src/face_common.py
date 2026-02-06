# src/face_common.py
from __future__ import annotations
import time
import os
import cv2
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

# Try imports
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except Exception as e:
    mp = None
    _MP_IMPORT_ERROR = e

# Import alignment from existing module
from .haar_5pt import align_face_5pt

# -------------------------
# Data Structures
# -------------------------
@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray # (5,2) float32 in FULL-frame coords
    ear: float = 0.3 # Average Eye Aspect Ratio (0.0=closed, ~0.3=open)

@dataclass
class ActionType(Enum):
    FACE_LOCKED = auto()
    FACE_LOST = auto()
    HEAD_LEFT = auto()
    HEAD_RIGHT = auto()
    EYE_BLINK = auto()
    SMILE = auto()

@dataclass
class Action:
    type: ActionType
    timestamp: float
    details: str = ""

@dataclass
class FaceLock:
    target_name: str
    target_emb: np.ndarray
    last_seen: float = field(default_factory=time.time)
    last_position: Optional[Tuple[float, float]] = None
    last_ear: Optional[float] = None
    history: List[Action] = field(default_factory=list)
    consecutive_frames: int = 0
    
    def update_position(self, kps: np.ndarray, current_ear: float) -> List[Action]:
        actions = []
        current_time = time.time()
        
        # Calculate face center
        center_x = kps[:, 0].mean()
        center_y = kps[:, 1].mean()
        
        # 1. Detect Head Turn (Yaw) using geometric ratios
        # kps: 0=LeftEye, 1=RightEye, 2=Nose
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        
        dist_l_n = np.linalg.norm(left_eye - nose)
        dist_r_n = np.linalg.norm(right_eye - nose)
        total_span = dist_l_n + dist_r_n + 1e-6
        yaw_ratio = dist_l_n / total_span
        
        # Tuned Thresholds for Mirror View
        # < 0.45: Looking Left (left eye dist smaller)
        # > 0.55: Looking Right (right eye dist smaller)
        if yaw_ratio < 0.45:
            actions.append(Action(ActionType.HEAD_LEFT, current_time, f"Turned Left (ratio: {yaw_ratio:.2f})"))
        elif yaw_ratio > 0.55:
            actions.append(Action(ActionType.HEAD_RIGHT, current_time, f"Turned Right (ratio: {yaw_ratio:.2f})"))
            
        # 2. Detect Smile (Robust: Mouth Width / Eye Distance)
        mouth_width = np.linalg.norm(kps[3] - kps[4])
        eye_dist_inter = np.linalg.norm(left_eye - right_eye)
        smile_ratio = mouth_width / (eye_dist_inter + 1e-6)
        
        # Smile threshold > 0.95
        if smile_ratio > 0.95: 
             actions.append(Action(ActionType.SMILE, current_time, f"Smile detected (r: {smile_ratio:.2f})"))

        # 3. Detect Blink (EAR)
        # EAR drops below threshold (~0.20) during blink
        if self.last_ear is not None:
             # Falling edge
             if current_ear < 0.20 and self.last_ear > 0.20:
                 actions.append(Action(ActionType.EYE_BLINK, current_time, f"Blink (EAR: {current_ear:.2f})"))
        
        # Update state
        self.last_position = (center_x, center_y)
        self.last_ear = current_ear
        self.last_seen = current_time
        self.consecutive_frames += 1
        
        return actions

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool

# -------------------------
# Math helpers
# -------------------------
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    return float(np.dot(a, b))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[int, int, int, int]:
    x1 = int(max(0, min(W - 1, round(x1))))
    y1 = int(max(0, min(H - 1, round(y1))))
    x2 = int(max(0, min(W - 1, round(x2))))
    y2 = int(max(0, min(H - 1, round(y2))))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def _bbox_from_5pt(kps: np.ndarray, pad_x: float = 0.55, pad_y_top: float = 0.85, pad_y_bot: float = 1.15) -> np.ndarray:
    k = kps.astype(np.float32)
    x_min = float(np.min(k[:, 0]))
    x_max = float(np.max(k[:, 0]))
    y_min = float(np.min(k[:, 1]))
    y_max = float(np.max(k[:, 1]))
    w = max(1.0, x_max - x_min)
    h = max(1.0, y_max - y_min)
    x1 = x_min - pad_x * w
    x2 = x_max + pad_x * w
    y1 = y_min - pad_y_top * h
    y2 = y_max + pad_y_bot * h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float) -> bool:
    k = kps.astype(np.float32)
    le, re, no, lm, rm = k
    eye_dist = float(np.linalg.norm(re - le))
    if eye_dist < float(min_eye_dist): return False
    if not (lm[1] > no[1] and rm[1] > no[1]): return False
    return True

def load_db_npz(db_path: Path) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    try:
        data = np.load(str(db_path), allow_pickle=True)
        out: Dict[str, np.ndarray] = {}
        for k in data.files:
            out[k] = np.asarray(data[k], dtype=np.float32).reshape(-1)
        return out
    except Exception as e:
        print(f"Warning: Failed to load database {db_path}: {e}. Starting with empty DB.")
        return {}

# -------------------------
# Embedder
# -------------------------
class ArcFaceEmbedderONNX:
    def __init__(
        self,
        model_path: str = "models/embedder_arcface.onnx",
        input_size: Tuple[int, int] = (112, 112),
        debug: bool = False,
    ):
        self.model_path = model_path
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        self.debug = bool(debug)
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

    def _preprocess(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        img = aligned_bgr_112
        if img.shape[1] != self.in_w or img.shape[0] != self.in_h:
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(v) + eps)
        return (v / n).astype(np.float32)

    def embed(self, aligned_bgr_112: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr_112)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = np.asarray(y, dtype=np.float32).reshape(-1)
        return self._l2_normalize(emb)

# -------------------------
# Multi-face Haar + FaceMesh(ROI) 5pt
# -------------------------
class HaarFaceMesh5pt:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        model_path: str = "models/face_landmarker.task",
        min_size: Tuple[int, int] = (70, 70),
        debug: bool = False,
    ):
        self.debug = bool(debug)
        self.min_size = tuple(map(int, min_size))
        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")
        
        if mp is None:
            raise RuntimeError(
                f"mediapipe import failed: {_MP_IMPORT_ERROR}\n"
                f"Install: pip install mediapipe"
            )
        
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found: {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for ROI processing
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # 5pt indices
        self.IDX_LEFT_EYE = 33
        self.IDX_RIGHT_EYE = 263
        self.IDX_NOSE_TIP = 1
        self.IDX_MOUTH_LEFT = 61
        self.IDX_MOUTH_RIGHT = 291
        
        # EAR indices (MediaPipe Mesh)
        # Left Eye: 33, 160, 158, 133, 153, 144
        self.EAR_L = [33, 160, 158, 133, 153, 144]
        # Right Eye: 362, 385, 387, 263, 373, 380
        self.EAR_R = [362, 385, 387, 263, 373, 380]

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        if faces is None or len(faces) == 0:
            return np.zeros((0, 4), dtype=np.int32)
        return faces.astype(np.int32) # (x,y,w,h)

    def _calc_ear(self, lm, idxs, W, H):
        pts = []
        for i in idxs:
            p = lm[i]
            pts.append(np.array([p.x * W, p.y * H]))
        
        # Vertical 1: p1-p5 (indices 1-5 in list of 6)
        v1 = np.linalg.norm(pts[1] - pts[5])
        # Vertical 2: p2-p4 (indices 2-4 in list of 6)
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal: p0-p3 (indices 0-3 in list of 6)
        h = np.linalg.norm(pts[0] - pts[3])
        
        if h < 1e-6: return 0.0
        return (v1 + v2) / (2.0 * h)

    def _roi_facemesh_5pt(self, roi_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        H, W = roi_bgr.shape[:2]
        if H < 20 or W < 20: return None, 0.0
        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)
        
        if not res.face_landmarks: return None, 0.0
        
        lm = res.face_landmarks[0]
        
        # 1. Extract 5pt
        idxs = [self.IDX_LEFT_EYE, self.IDX_RIGHT_EYE, self.IDX_NOSE_TIP, self.IDX_MOUTH_LEFT, self.IDX_MOUTH_RIGHT]
        pts = []
        for i in idxs:
            p = lm[i]
            pts.append([p.x * W, p.y * H])
        kps = np.array(pts, dtype=np.float32)
        if kps[0, 0] > kps[1, 0]: kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]: kps[[3, 4]] = kps[[4, 3]]
        
        # 2. Extract EAR
        ear_l = self._calc_ear(lm, self.EAR_L, W, H)
        ear_r = self._calc_ear(lm, self.EAR_R, W, H)
        avg_ear = (ear_l + ear_r) / 2.0
        
        return kps, avg_ear

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)
        if faces.shape[0] == 0: return []
        
        # sort by area desc, keep top max_faces
        areas = faces[:, 2] * faces[:, 3]
        order = np.argsort(areas)[::-1]
        faces = faces[order][:max_faces]
        
        out: List[FaceDet] = []
        for (x, y, w, h) in faces:
            mx, my = 0.25 * w, 0.35 * h
            rx1, ry1, rx2, ry2 = _clip_xyxy(x - mx, y - my, x + w + mx, y + h + my, W, H)
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            kps_roi, ear_val = self._roi_facemesh_5pt(roi)
            if kps_roi is None: continue
            
            kps = kps_roi.copy()
            kps[:, 0] += float(rx1)
            kps[:, 1] += float(ry1)
            
            if not _kps_span_ok(kps, min_eye_dist=max(10.0, 0.18 * float(w))): continue
            
            bb = _bbox_from_5pt(kps, pad_x=0.55, pad_y_top=0.85, pad_y_bot=1.15)
            x1, y1, x2, y2 = _clip_xyxy(bb[0], bb[1], bb[2], bb[3], W, H)
            
            out.append(FaceDet(
                x1=x1, y1=y1, x2=x2, y2=y2, 
                score=1.0, kps=kps.astype(np.float32),
                ear=ear_val
            ))
        return out

    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()

# -------------------------
# Matcher
# -------------------------
class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = float(dist_thresh)
        self._names: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self._rebuild()

    def _rebuild(self):
        self._names = sorted(self.db.keys())
        if self._names:
            self._mat = np.stack([self.db[n].reshape(-1).astype(np.float32) for n in self._names], axis=0)
        else:
            self._mat = None

    def reload_from(self, path: Path):
        self.db = load_db_npz(path)
        self._rebuild()

    def match(self, emb: np.ndarray) -> MatchResult:
        if self._mat is None or len(self._names) == 0:
            return MatchResult(name=None, distance=1.0, similarity=0.0, accepted=False)
        e = emb.reshape(1, -1).astype(np.float32)
        sims = (self._mat @ e.T).reshape(-1)
        best_i = int(np.argmax(sims))
        best_sim = float(sims[best_i])
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh
        return MatchResult(
            name=self._names[best_i] if ok else None,
            distance=float(best_dist),
            similarity=float(best_sim),
            accepted=bool(ok),
        )

def save_action_history(face_name: str, actions: List[Action]):
    if not actions:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{face_name}_history_{timestamp}.txt"
    os.makedirs("logs", exist_ok=True)
    
    with open(f"logs/{filename}", "w") as f:
        for action in actions:
            time_str = datetime.fromtimestamp(action.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")
            f.write(f"{time_str} - {action.type.name}: {action.details}\n")
