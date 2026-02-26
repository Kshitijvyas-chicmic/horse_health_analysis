from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import subprocess
import os
import json

router = APIRouter()

class AutomationTask(BaseModel):
    task: str

@router.api_route("/run-task", methods=["GET", "POST"])
async def run_automation_task(request: Request, task: str = None):
    # Standardize extraction: check query param first, then body
    task_name = task
    
    if not task_name and request.method == "POST":
        try:
            body = await request.json()
            task_name = body.get("task")
        except:
            pass
            
    if not task_name:
        raise HTTPException(status_code=400, detail="No task specified in query or body")
        
    task = task_name
    cwd = "/app"
    
    commands = {
        "split": [
            "python3", "split_dataset.py",
            "--input", "data/annotations/cvat/person_keypoints_default_590.json",
            "--train_out", "datasets/train.json",
            "--val_out", "datasets/val.json"
        ],
        "fix_bbox": [
            "bash", "-c",
            "python3 fix_bbox_from_keypoints.py datasets/train.json datasets/train_fixed.json && python3 fix_bbox_from_keypoints.py datasets/val.json datasets/val_fixed.json"
        ],
        "validate": [
            "python3", "tools/validate_data_integrity.py",
            "--train", "datasets/train_fixed.json",
            "--val", "datasets/val_fixed.json"
        ],
        "update_config": [
            "python3", "tools/update_training_config.py",
            "--config", "mmpose/custom_configs/rtmpose_hoof_4kp.py",
            "--train", "datasets/train_fixed.json",
            "--val", "datasets/val_fixed.json"
        ],
        "train": [
            "docker", "exec", "horse_health_trainer",
            "bash", "-c", "python3 mmpose/tools/train.py mmpose/custom_configs/rtmpose_hoof_4kp.py > /proc/1/fd/1 2>&1"
        ]
    }

    if task not in commands:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}")

    try:
        # For training, we might want to run it in background, but for dry-run
        # let's wait for the others to get the output.
        if task == "train":
            # For training, we use Popen and don't wait to avoid timeout
            process = subprocess.Popen(commands[task], cwd=cwd)
            return {"status": "started", "task": task, "pid": process.pid}
        
        result = subprocess.run(
            commands[task],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "status": "success",
            "task": task,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "task": task,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "error": str(e)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
