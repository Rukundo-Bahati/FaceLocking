# Face Recognition with Real-time Tracking

A real-time face recognition system with face locking capabilities, designed to identify and track specific individuals while monitoring their actions.

## Core Features
- **Face Recognition**: 
  - Real-time face detection and identification
  - Multi-person support
  - High accuracy with ArcFace embeddings

- **Face Locking**:
  - Track specific individuals in real-time
  - Monitor head movements (left/right)
  - Detect smiles and facial expressions
  - Automatic action logging with timestamps

## Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Models**:
   Ensure the following models are in the `models/` directory:
   - `embedder_arcface.onnx` (ArcFace recognition)
   - `face_landmarker.task` (MediaPipe FaceMesh)

## Usage

### 1. Enrollment
Enroll new identities by capturing face samples from the camera.
```bash
python -m src.enroll
```
- **Controls**:
  - `SPACE`: Capture a single sample.
  - `a`: Toggle auto-capture.
  - `s`: Save enrollment to database.
  - `q`: Quit.

### 2. Recognition
Run real-time multi-face recognition with face locking and action tracking.
```bash
python -m src.recognize
```
- **Controls**:
  - `+/-`: Adjust distance threshold live.
  - `r`: Reload database from disk.
  - `d`: Toggle debug overlay.
  - `l`: Lock/unlock the currently recognized face.
  - `q`: Quit.

#### Face Locking Features
- Lock onto a specific person by pressing `l` when their face is detected
- The system will track the locked face and log their actions
- Actions include:
  - Head movements (left/right)
  - Smile detection
  - Face locking/unlocking events
- All actions are timestamped and saved to `logs/[Name]_history_[timestamp].txt`

### 3. Evaluation
Evaluate the model's performance on enrolled crops and find the optimal threshold.
```bash
python -m src.evaluate
```

### 4. Demos
Visualize embeddings or detection/landmarks:
```bash
python -m src.embed    # Embedding heatmap visualization
python -m src.haar_5pt  # Detection and landmark visualization
```

## Face Locking System

### How Face Locking Works
1. **Activation**: 
   - Press 'l' when a recognized face is on screen
   - The system will lock onto the closest recognized face
   - A visual indicator (orange border) shows the locked face

2. **Tracking**:
   - The system continues to track the locked face even if other faces appear
   - If the face is temporarily lost, the system will try to reacquire it for 2 seconds
   - The lock is automatically released if the face is not found after this period

3. **Actions Detected**:
   - **Head Movements**:
     - Left/Right: Tracks horizontal head rotation
     - Movement threshold: 10 pixels (adjustable)
   - **Facial Expressions**:
     - Smile detection: Measures mouth corner movement
     - Smile threshold: Ratio > 1.1 (configurable)
   - **System Events**:
     - Face locked/unlocked
     - Face lost/regained

### History Files
- **Location**: All logs are saved in the `logs/` directory
- **Naming Convention**: `[Name]_history_[timestamp].txt`
  - Example: `Bahati_history_20260131132049.txt`
- **Log Format**:
  ```
  [YYYY-MM-DD HH:MM:SS.microseconds] - ACTION_TYPE: Description
  ```
- **Example Entries**:
  ```
  2026-01-31 13:20:11.324267 - FACE_LOCKED: Face locked: Bahati
  2026-01-31 13:20:20.225528 - HEAD_RIGHT: Moved right by 31.9px
  2026-01-31 13:20:21.684933 - SMILE: Smile detected (ratio: 14.97)
  ```
- **Log Management**:
  - New log file created for each recognition session
  - Timestamp in filename helps track different usage sessions
  - Logs are automatically created when face locking is used

### Technical Details
- **Face Recognition**: Uses ArcFace for generating unique face embeddings
- **Landmark Tracking**: MediaPipe FaceMesh tracks 5 key facial points
- **Performance**: Optimized for real-time processing on CPU
