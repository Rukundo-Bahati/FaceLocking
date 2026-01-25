# Face Recognition 5-Point Pipeline

A robust, CPU-friendly face recognition system using Haar Cascades, MediaPipe FaceMesh (5-point landmarks), and ArcFace embeddings.

## Features
- **Fast Detection**: Uses Haar Cascades for initial face localization.
- **Stable Landmarks**: Uses MediaPipe FaceMesh to extract 5 keypoints (eyes, nose, mouth corners).
- **Accurate Embeddings**: Uses ArcFace (InsightFace) ONNX model for high-quality face vectors.
- **Multi-Face Support**: Real-time recognition of multiple people.
- **Evaluation Tools**: Built-in threshold tuning and performance metrics.

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
Run real-time multi-face recognition.
```bash
python -m src.recognize
```
- **Controls**:
  - `+/-`: Adjust distance threshold live.
  - `r`: Reload database from disk.
  - `d`: Toggle debug overlay.
  - `q`: Quit.

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

## Technical Details
The pipeline follows these steps:
1. **Detection**: Haar Cascade finds face ROIs.
2. **Landmarks**: MediaPipe FaceMesh extracts 5 keypoints within the ROI.
3. **Alignment**: Similarity transform aligns the face to a 112x112 template.
4. **Embedding**: ArcFace ONNX generates a 512-dimensional L2-normalized vector.
5. **Matching**: Cosine distance (1 - dot product) is compared against a threshold.
