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

# Update core build tools
pip install --upgrade pip setuptools wheel
```

#### Option A: Production/Inference (Lightweight - CPU Only)
Recommended for your 4-core production server. This version is small and fast to install.
```bash
pip install -r requirements.txt
```

#### Option B: Development/Training (Full - GPU Required)
Required only for training new models on a GPU-enabled machine. 
*Note: This requires the CUDA index URL for PyTorch.*
```bash
pip install -r requirements-train.txt --find-links https://download.pytorch.org/whl/torch_stable.html
```

> [!TIP]
> **If you encounter build errors (like "chumpy" or "xtcocotools"):**
> The lightweight `requirements.txt` removes `chumpy` as it's not needed for inference. If you still see errors in the full setup, try installing the package individually with `pip install --no-build-isolation <package_name>`.

### 3. Model Dependencies
Ensure `mmpose` is available as a submodule and the checkpoints exist in `mmpose/work_dirs/`.

## Running the Server

### Local Development
```bash
uvicorn apis.main:app --host 0.0.0.0 --port 8001
```

### Production (Recommended)
For production servers (like your HTTPS setup), use **Gunicorn** with Uvicorn workers. This allows for multiple worker processes and better stability.

```bash
# Kill any existing process on 8000
fuser -k 8000/tcp

# Start with Gunicorn (4 workers recommended for most servers)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker apis.main:app --bind 0.0.0.0:8000 --access-log - --error-log -
```

*Note: If Gunicorn is not available, you can still use Uvicorn with proxy flags:*
```bash
uvicorn apis.main:app --host 0.0.0.0 --port 8000 --proxy-headers --forwarded-allow-ips='*'
```

### Server Lifecycle
- **Startup**: The server initializes the `HPAPredictor` class, which loads the RTMPose model into memory. This ensures subsequent requests are processed quickly.
- **Concurrency**: Network I/O (fetching images from S3) is handled asynchronously. Model inference is handled sequentially to manage CPU/Memory resources effectively under load.

## Interactive Documentation (Write Access)

You can interactively test the API endpoints (send "write" requests) using the built-in Swagger UI:

🔗 **[Interactive API Docs](https://horse-health.projectlabs.in/docs)**

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

## Production Deployment Guide

To deploy this API on a fresh client server, follow these steps:

### 1. Model File Sync
The model checkpoints are large and may not be in Git. Ensure the following directory is physically present on the server:
- `mmpose/work_dirs/rtmpose_hoof_unified_jan12/` (Contains `epoch_300.pth` and the `.py` config)

### 2. Process Management (Gunicorn)
For production, use **Gunicorn** with **Uvicorn workers** for better stability and process management:

```bash
conda activate env_mmpose_env
pip install gunicorn
# Run with 4 workers (or more depending on CPU cores)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker apis.main:app --bind 0.0.0.0:8000 --timeout 120
```

### 3. Using Systemd (Auto-restart)
To ensure the API starts on boot and restarts after crashes, create a systemd service:

```ini
# /etc/systemd/system/horse-api.service
[Unit]
Description=Horse Health Analysis API
After=network.target

[Service]
User=your_user
WorkingDirectory=/path/to/horse_health_analysis
ExecStart=/path/to/miniconda3/envs/env_mmpose_env/bin/gunicorn apis.main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001 --timeout 120
Restart=always

[Install]
WantedBy=multi-user.target
```

### 4. Firewall & Port Access
Ensure port **8001** is open in the server's firewall (AWS Security Group, UFW, or IPTables) to allow the Node.js/Mobile client to communicate with it.

### 5. CORS Security
Currently, CORS is set to `allow_origins=["*"]`. In production, you should update `apis/main.py` to only allow the specific domain of your web/mobile app.

## Troubleshooting

### "Address already in use" (Port 8001)
If you see the error `[Errno 98] error while attempting to bind on address`, it means another process is already using port 8001. You can clear it using:
```bash
fuser -k 8001/tcp
```
*Note: You may need `sudo` depending on your server permissions.*

## Production Deployment Checklist (Step-by-Step)

Follow this order to ensure a smooth transition to your client's production server:

1. **Clone the Repository (with Submodules)**:
   > [!IMPORTANT]
   > `mmpose` is a Git Submodule. You MUST use the `--recursive` flag or the folder will be empty.
   ```bash
   git clone --recursive https://github.com/Kshitijvyas-chicmic/horse_health_analysis.git
   cd horse_health_analysis
   ```
   *If you already cloned without submodules, run:*
   `git submodule update --init --recursive`

2. **Sync Model Weights (Manual Step)**:
   > [!WARNING]
   > Model weights (`.pth` files) are EXCLUDED from Git to keep the repo size small.
   - **Step A**: On the development server, go to `mmpose/work_dirs/`.
   - **Step B**: Zip or SCP the folder to the production server.
   - **Step C**: Place it at `horse_health_analysis/mmpose/work_dirs/rtmpose_hoof_unified_jan12/`.

3. **Setup Environment**:
   ```bash
   conda create -n env_mmpose_env python=3.10 -y
   conda activate env_mmpose_env
   pip install -r requirements.txt
   # gunicorn is now included in requirements.txt
   ```

3. **Sync Model Weights**:
   > [!IMPORTANT]
   > Model weights (`.pth` files) are NOT in Git. You must manually copy the `mmpose/work_dirs/` folder from this server to the production server at the exact same path.

4. **Firewall & Port**:
   - Open port **8000** for incoming TCP traffic.
   - If using a reverse proxy (like Nginx), you might use port 80/443 pointing to 8000.

5. **Security Update**:
   - Open `apis/main.py`.
   - Update `allow_origins = ["*"]` to `allow_origins = ["https://your-client-app.com"]`.

6. **Go Live**:
   - Start via Systemd (recommended) or use the Gunicorn command from the "Process Management" section above.

---
*Last updated: Feb 5, 2026*
