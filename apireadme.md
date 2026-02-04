# Horse Health Analysis API Documentation

This API provides tools for analyzing horse limb health from images, specifically focusing on Hoof-Pastern Axis (HPA) metrics.

## Setup & Installation

### 1. Prerequisites
- Python 3.10
- Conda (Miniconda/Anaconda)
- NVIDIA GPU (optional but recommended for training; inference runs on CPU by default)

### 2. Environment Setup
```bash
conda create -n env_mmpose_env python=3.10 -y
conda activate env_mmpose_env
pip install -r requirements.txt
```

### 3. Model Dependencies
Ensure `mmpose` is available as a submodule and the checkpoints exist in `mmpose/work_dirs/`.

## Running the Server

Start the FastAPI server using `uvicorn`:
```bash
uvicorn apis.main:app --host 0.0.0.0 --port 8001
```

### Server Lifecycle
- **Startup**: The server initializes the `HPAPredictor` class, which loads the RTMPose model into memory. This ensures subsequent requests are processed quickly.
- **Concurrency**: Network I/O (fetching images from S3) is handled asynchronously. Model inference is handled sequentially to manage CPU/Memory resources effectively under load.

## Interactive Documentation (Write Access)

You can interactively test the API endpoints (send "write" requests) using the built-in Swagger UI:

🔗 **[Interactive API Docs](http://192.180.3.178:8001/docs)**

Here you can use the **"Try it out"** button to upload images or send batch URLs and see immediate results.

## API Endpoints

### 1. Single Image Upload (Multipart)
`POST /api/v1/batch-analyze` (Note: Also available at legacy `/analyze`)

### 2. Batch Analysis (S3 URLs)
`POST /api/v1/batch-analyze`

**Request Body:**
```json
{
  "images": [
    {
      "image_id": "image_001",
      "url": "https://s3.amazonaws.com/bucket/image1.jpg"
    }
  ]
}
```

**Response Body:**
```json
{
  "results": [
    {
      "image_id": "image_001",
      "metrics": {
        "success": true,
        "best_zone": "Floor-Scan",
        "pastern_angle": 53.5,
        "hoof_angle": 50.2,
        "hpa_dev": 3.3,
        "image_base64": "...",
        "error": null
      }
    }
  ]
}
```

### 3. Advanced Multi-Leg Analysis (V2)
`POST /api/v2/analyze`

This endpoint implements a modular business logic agent that processes up to 4 legs, calculates deviation-based scores, and generates clinical notes.

**Request Body:**
```json
{
  "frontLeftLateral": "<base64 | url | null>",
  "frontRightLateral": "<base64 | url | null>",
  "backLeftLateral": "<base64 | url | null>",
  "backRightLateral": "<base64 | url | null>"
}
```

**Response Body:**
```json
{
  "frontLeftScanScore": 8.0,
  "frontLeftNotes": "Horse seems HEALTHY!",
  "frontLeftQuality": 10,
  "...": "...",
  "scanScore": 8.0,
  "notes": "Healthy",
  "status": 1,
  "scanId": "#SCN001"
}
```

## Scalability & Production Standards
- **Modular Architecture**: V2 logic is split into stateless services for scoring, notes, quality, and aggregation.
- **Strict Contract**: All keys are guaranteed in the response, using `null` for missing inputs.
- **Resource Management**: Sequential inference ensures the server remains responsive even under batch load.
