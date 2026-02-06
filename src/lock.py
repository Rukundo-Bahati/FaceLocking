# src/lock.py
"""
Face Locking Script
Dedicated script for locking onto a specific person and tracking them extensively.
When locked, this script actively ignores all other faces to prevent "jumping".

Usage:
    python -m src.lock
    python -m src.lock --name "PersonName"

Keys:
    q : Quit
    r : Reload DB
    l : Lock/Unlock (if a face is selected)
"""
import argparse
import time
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List

from .face_common import (
    FaceDet, FaceLock, Action, ActionType, MatchResult,
    HaarFaceMesh5pt, ArcFaceEmbedderONNX, FaceDBMatcher,
    load_db_npz, align_face_5pt, save_action_history
)

def main():
    parser = argparse.ArgumentParser(description="Face Locking System")
    parser.add_argument("--name", type=str, help="Auto-lock onto this person on startup")
    args = parser.parse_args()

    # Paths
    db_path = Path("data/db/face_db.npz")
    os.makedirs("logs", exist_ok=True)

    # Initialize components
    try:
        det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
        embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx")
        db = load_db_npz(db_path)
        matcher = FaceDBMatcher(db=db, dist_thresh=0.34)
    except Exception as e:
        print(f"Error initializing components: {e}")
        return

    # Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not available")
        det.close()
        return

    print("=== Face Lock System ===")
    print("Keys: 'q'=quit, 'l'=lock/unlock current face, 'r'=reload DB")
    if args.name:
        print(f"Waiting to lock onto: {args.name}")

    # State
    face_lock: Optional[FaceLock] = None
    last_potential_target: Optional[tuple] = None # (name, emb, kps)
    
    # Tracking parameters
    MAX_TRACK_DIST = 200  # Increased for robustness
    
    t0 = time.time()
    frames = 0
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            
            # FPS calculation
            frames += 1
            if time.time() - t0 >= 1.0:
                fps = frames / (time.time() - t0)
                frames = 0
                t0 = time.time()

            vis = frame.copy()
            H, W = vis.shape[:2]
            current_time = time.time()

            # 1. Detection
            all_faces = det.detect(frame, max_faces=5)
            
            # 2. Logic Split: Locked vs Unlocked
            if face_lock:
                # --- LOCKED MODE ---
                # NEVER unlock automatically. Keep searching until found.
                
                best_face = None
                
                # A. Spatial Check (Fastest, prevents jumping)
                candidates = []
                for f in all_faces:
                    if face_lock.last_position:
                        center_x = f.kps[:, 0].mean()
                        center_y = f.kps[:, 1].mean()
                        dist = np.hypot(center_x - face_lock.last_position[0], 
                                      center_y - face_lock.last_position[1])
                        
                        # If very close, assume it's the target (unless we have reason to doubt)
                        if dist < MAX_TRACK_DIST:
                            # Score based on distance (closer is better)
                            score = 1.0 - (dist / MAX_TRACK_DIST)
                            candidates.append((score, f))
                
                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    best_face = candidates[0][1]
                
                # B. Identity Re-acquisition (Fallback if spatial fails or lost)
                # If we didn't find them spatially (detected faces are too far),
                # we MUST scan all faces to find them again.
                if not best_face:
                    for f in all_faces:
                        aligned, _ = align_face_5pt(frame, f.kps)
                        emb = embedder.embed(aligned)
                        mr = matcher.match(emb)
                        if mr.accepted and mr.name == face_lock.target_name:
                            best_face = f
                            # Update reference embedding to handle lighting changes? 
                            # Maybe risky if we match an imposter, but ok for now.
                            break

                if best_face:
                    target_face = best_face
                    
                    # Update lock state
                    # Update lock state
                    actions = face_lock.update_position(target_face.kps, target_face.ear)
                    face_lock.history.extend(actions)
                    for action in actions:
                        print(f"[Action] {action.type.name}: {action.details}")
                        
                    # Visuals for Locked Face (Strict: ONLY this face)
                    cv2.rectangle(vis, (target_face.x1, target_face.y1), (target_face.x2, target_face.y2), (255, 165, 0), 3)
                    cv2.putText(vis, f"LOCKED: {face_lock.target_name}", 
                               (target_face.x1, target_face.y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                    
                    # Reset last seen
                    face_lock.last_seen = current_time

            else:
                # --- UNLOCKED MODE ---
                # Two sub-modes: 
                # A. Strict Search (if args.name is set but not found yet) -> Hide others
                # B. General Scan (if no args.name) -> Show all
                
                if args.name:
                    # STRICT SEARCH MODE
                    found_target = False
                    for f in all_faces:
                        aligned, _ = align_face_5pt(frame, f.kps)
                        emb = embedder.embed(aligned)
                        mr = matcher.match(emb)
                        
                        if mr.accepted and mr.name == args.name:
                            # FOUND! Auto-lock immediately
                            face_lock = FaceLock(target_name=mr.name, target_emb=emb)
                            face_lock.update_position(f.kps, f.ear)
                            print(f"[FaceLock] Auto-locked onto {mr.name}")
                            found_target = True
                            break # Go to loop start to render locked state next frame
                        
                        # Use embedding similarities to find 'best' candidates if not accepted?
                        # For now, just show nothing for others.
                    
                    if not found_target:
                         # Draw user feedback
                         cv2.putText(vis, f"WAITING FOR: {args.name}", (10, H - 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                         
                else:
                    # GENERAL SCAN MODE
                    for f in all_faces:
                        aligned, _ = align_face_5pt(frame, f.kps)
                        emb = embedder.embed(aligned)
                        mr = matcher.match(emb)
                        
                        color = (0, 255, 0) if mr.accepted else (0, 0, 255)
                        label = mr.name if mr.accepted else "Unknown"
                        
                        # Draw
                        cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                        cv2.putText(vis, f"{label}", (f.x1, f.y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        # Store as potential lock target
                        if mr.accepted:
                            last_potential_target = (mr.name, emb, f.kps, f.ear)
                            hint = f"Press 'l' to lock {mr.name}"
                            cv2.putText(vis, hint, (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # --- HUD & INFO DISPLAY ---
            
            # 1. Top Bar Background
            cv2.rectangle(vis, (0, 0), (W, 40), (0, 0, 0), -1)
            
            # FPS
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Status Center
            if face_lock:
                 status_text = f"LOCKED: {face_lock.target_name}"
                 # Check if recently seen
                 if (current_time - face_lock.last_seen) > 0.5:
                     status_text += " (SEARCHING...)"
                     status_color = (0, 0, 255) # Red text if searching
                 else:
                     status_color = (0, 255, 255) # Yellow/Orange if tracking
                 
                 tsfc = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                 tx = (W - tsfc[0]) // 2
                 cv2.putText(vis, status_text, (tx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                 
                 # --- ACTION HISTORY SIDEBAR ---
                 # Draw a semi-transparent box on the right side
                 side_w = 300
                 overlay = vis.copy()
                 cv2.rectangle(overlay, (W - side_w, 40), (W, 300), (0, 0, 0), -1)
                 vis = cv2.addWeighted(overlay, 0.6, vis, 0.4, 0)
                 
                 cv2.putText(vis, "ACTION HISTORY:", (W - side_w + 10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                 
                 if face_lock.history:
                     # Show last 8 actions
                     recent = face_lock.history[-8:]
                     y_off = 90
                     for act in reversed(recent):
                         t_str = datetime.fromtimestamp(act.timestamp).strftime("%H:%M:%S")
                         # Shorten details
                         detail = act.details
                         if len(detail) > 25: detail = detail[:22] + "..."
                         
                         line = f"[{t_str}] {act.type.name}"
                         cv2.putText(vis, line, (W - side_w + 10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                         cv2.putText(vis, f"  {detail}", (W - side_w + 10, y_off + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                         y_off += 35

            else:
                 status_text = "SCANNING"
                 tsfc = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                 tx = (W - tsfc[0]) // 2
                 cv2.putText(vis, status_text, (tx, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Face Lock Guard", vis)
            
            # Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                matcher.reload_from(db_path)
                print("DB Reloaded")
            elif key == ord('l'):
                if face_lock:
                    print(f"[FaceLock] Manual Unlock: {face_lock.target_name}")
                    save_action_history(face_lock.target_name, face_lock.history)
                    face_lock = None
                elif last_potential_target:
                    name, emb, kps, ear = last_potential_target
                    face_lock = FaceLock(target_name=name, target_emb=emb)
                    face_lock.update_position(kps, ear)
                    print(f"[FaceLock] Manual Lock: {name}")
                    last_potential_target = None
                else:
                    print("[FaceLock] No recognized face to lock onto!")

    finally:
        det.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
