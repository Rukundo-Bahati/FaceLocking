# src/recognize.py
"""
Multi-face recognition (CPU-friendly) using shared pipeline.
This script focuses PURELY on recognition (labeling faces).
Locking functionality has been moved to `src/lock.py`.

Run:
    python -m src.recognize

Keys:
    q : quit
    r : reload DB from disk
    +/- : adjust threshold (distance) live
    d : toggle debug overlay
"""
from __future__ import annotations
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

# Import shared components from face_common
from .face_common import (
    HaarFaceMesh5pt,
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    load_db_npz,
    align_face_5pt
)

def main():
    parser = argparse.ArgumentParser(description="Real-time Face Recognition (No Locking)")
    args = parser.parse_args()
    
    db_path = Path("data/db/face_db.npz")
    
    # Initialize shared components
    try:
        det = HaarFaceMesh5pt(
            min_size=(70, 70),
            debug=False,
        )
        embedder = ArcFaceEmbedderONNX(
            model_path="models/embedder_arcface.onnx",
            input_size=(112, 112),
            debug=False,
        )
        db = load_db_npz(db_path)
        matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
    except Exception as e:
        print(f"Error initializing components: {e}")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")
        det.close()
        return
    
    print("=== Pure Recognition ===")
    print("Keys: q=quit, r=reload DB, +/- threshold, d=debug overlay")

    t0 = time.time()
    frames = 0
    fps: Optional[float] = None
    show_debug = False
    
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # Detection using shared component
            faces = det.detect(frame, max_faces=5)
            vis = frame.copy()
            
            # FPS calculation
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                frames = 0
                t0 = time.time()
            
            # Visualization Setup
            h, w = vis.shape[:2]
            thumb = 112
            pad = 8
            x0 = w - thumb - pad
            y0 = 80
            shown = 0
            
            for i, f in enumerate(faces):
                # Basic Draw
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for (x, y) in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # Recognition (Align -> Embed -> Match)
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)
                
                # Labeling
                label = mr.name if mr.name is not None else "Unknown"
                dist_str = f"d={mr.distance:.2f}"
                
                # Color code: Green = Accepted Match, Red = Unknown
                color = (0, 255, 0) if mr.accepted else (0, 0, 255)
                
                # Text
                cv2.putText(vis, label, (f.x1, max(0, f.y1 - 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(vis, dist_str, (f.x1, max(0, f.y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Thumbnail preview (optional, kept from original)
                if y0 + thumb <= h and shown < 4:
                    vis[y0:y0 + thumb, x0:x0 + thumb] = aligned
                    cv2.putText(
                        vis,
                        f"{i+1}:{label}",
                        (x0, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )
                    y0 += thumb + pad
                    shown += 1
                
                if show_debug:
                    dbg = f"kpsLeye=({f.kps[0,0]:.0f},{f.kps[0,1]:.0f})"
                    cv2.putText(vis, dbg, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # UI Header
            header = f"IDs: {len(matcher._names)} | Thresh: {matcher.dist_thresh:.2f}"
            if fps is not None:
                header += f" | FPS: {fps:.1f}"
            cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "Mode: RECOGNIZE ONLY", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Face Recognition (Pure)", vis)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break
            elif key == ord("r"):
                matcher.reload_from(db_path)
                print(f"[recognize] reloaded DB: {len(matcher._names)} identities")
            elif key in (ord("+"), ord("=")):
                matcher.dist_thresh = float(min(1.20, matcher.dist_thresh + 0.01))
                print(f"[recognize] thresholds increased: {matcher.dist_thresh:.2f}")
            elif key == ord("-"):
                matcher.dist_thresh = float(max(0.05, matcher.dist_thresh - 0.01))
                print(f"[recognize] thresholds decreased: {matcher.dist_thresh:.2f}")
            elif key == ord("d"):
                show_debug = not show_debug
                print(f"[recognize] debug overlay: {'ON' if show_debug else 'OFF'}")

    finally:
        det.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()