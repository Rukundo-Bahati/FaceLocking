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
    cap = cv2.VideoCapture(1)
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
    # Tracking parameters
    MAX_TRACK_DIST = 200  # Increased for robustness
    REC_INTERVAL = 0.5    # Seconds between full ID scans
    last_rec_time = 0.0
    cached_others = []    # List of {'center':(x,y), 'mr':MatchResult}
    
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

            # 1. Detection & Filtering ("Ignoring Background Noise")
            raw_faces = det.detect(frame, max_faces=10)
            all_faces = []
            MIN_FACE_SIZE = 60 # Ignore shadows/noise smaller than this
            for f in raw_faces:
                w = f.x2 - f.x1
                h = f.y2 - f.y1
                if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
                    all_faces.append(f)
            
            # Decisions for this frame
            do_recognition = False
            if (current_time - last_rec_time) > REC_INTERVAL:
                do_recognition = True
                
            # Maps for drawing (id(face) -> (MatchResult, Embedding))
            face_identities = {} 
            target_face = None

            # 2. "Following the Box" (Tracking the Locked Person)
            if face_lock and face_lock.last_position:
                # Search for target spatially
                best_dist = float('inf')
                best_f = None
                
                for f in all_faces:
                    # Calculate distance
                    center_x = f.kps[:, 0].mean()
                    center_y = f.kps[:, 1].mean()
                    dist = np.hypot(center_x - face_lock.last_position[0], 
                                  center_y - face_lock.last_position[1])
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_f = f
                
                # If close enough, we TRUST it is the target (No Recalc)
                if best_f and best_dist < MAX_TRACK_DIST:
                    target_face = best_f
                    # Identify implicitly without running embedding
                else:
                    # Target lost (sticker fell off) or jumped too far
                    do_recognition = True

            # 3. "Skipping Heavy Work" (Identity Management)
            if do_recognition:
                # --- HEAVY FRAME (Run AI Brain) ---
                cached_others = [] # Clear cache
                
                for f in all_faces:
                    # Skip embedding the target if we are successfully tracking
                    if f is target_face:
                        continue
                    
                    # Embed & Match
                    aligned, _ = align_face_5pt(frame, f.kps)
                    emb = embedder.embed(aligned)
                    mr = matcher.match(emb)
                    face_identities[id(f)] = (mr, emb)
                    
                    # Check if this strictly searches for target (Re-acquisition)
                    if face_lock and not target_face:
                        if mr.accepted and mr.name == face_lock.target_name:
                            target_face = f # Found them again!
                            
                    # Cache for next frames
                    cx, cy = f.kps[:,0].mean(), f.kps[:,1].mean()
                    cached_others.append({'center': (cx, cy), 'mr': mr, 'emb': emb})
                
                last_rec_time = current_time
                
            else:
                # --- LIGHT FRAME (Use Cache) ---
                # Map non-target faces to cached identities
                for f in all_faces:
                    if f is target_face: continue
                    
                    # Find nearest cached identity
                    cx, cy = f.kps[:,0].mean(), f.kps[:,1].mean()
                    
                    best_match = None
                    best_emb = None
                    min_cache_dist = MAX_TRACK_DIST
                    
                    for item in cached_others:
                        dist = np.hypot(cx - item['center'][0], cy - item['center'][1])
                        if dist < min_cache_dist:
                            min_cache_dist = dist
                            best_match = item['mr']
                            best_emb = item['emb']
                            
                    if best_match:
                        face_identities[id(f)] = (best_match, best_emb)

            # 4. Final Processing & Visualization
            for f in all_faces:
                # Is this the locked target?
                if f is target_face:
                    # Update Lock State
                    if face_lock:
                         actions = face_lock.update_position(f.kps, f.ear)
                         face_lock.history.extend(actions)
                         face_lock.last_seen = current_time
                         
                         for action in actions:
                             print(f"[Action] {action.type.name}: {action.details}")

                    # Visuals (Orange)
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (255, 165, 0), 3)
                    label = face_lock.target_name if face_lock else "LOCKED"
                    cv2.putText(vis, f"LOCKED: {label}", 
                               (f.x1, f.y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                               
                else:
                    # Other faces
                    identity_data = face_identities.get(id(f))
                    mr, face_emb = identity_data if identity_data else (None, None)
                    
                    if mr and mr.accepted:
                        color = (0, 255, 0)
                        label = mr.name
                        
                        # Auto-lock logic check (only if UNLOCKED mode)
                        if not face_lock and args.name and mr.name == args.name:
                            # Auto-lock trigger
                            # Use the embedding we have (either fresh or cached)
                            face_lock = FaceLock(target_name=mr.name, target_emb=face_emb) 
                            face_lock.update_position(f.kps, f.ear)
                            print(f"[FaceLock] Auto-locked onto {mr.name}")
                            target_face = f 
                            # Continue to visualize as "scanning" for this frame, locked next frame
                            
                    elif mr:
                        color = (0, 0, 255)
                        label = "Unknown"
                    else:
                        color = (100, 100, 100)
                        label = "Scanning..."
                        
                    cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                    cv2.putText(vis, label, (f.x1, f.y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Interactive Lock Hint
                    if not face_lock and mr and mr.accepted:
                         last_potential_target = (mr.name, face_emb, f.kps, f.ear)
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
