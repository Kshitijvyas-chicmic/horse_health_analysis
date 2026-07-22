import asyncio
import logging
import httpx
import json
import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any
from apis.v5.schemas import AsyncScanRequest, WebhookPayload, AdvancedScanResponseV5
from apis.v5.routes import run_full_scan_logic

# The global in-memory queue
_scan_queue: asyncio.Queue = asyncio.Queue()
_worker_task: asyncio.Task = None

@dataclass
class ScanJob:
    job_id: str
    request: AsyncScanRequest
    predictor: Any  # HPAPredictor instance


async def enqueue_scan_job(request: AsyncScanRequest, predictor: Any, job_id: str):
    """Pushes a scan job to the queue."""
    job = ScanJob(job_id=job_id, request=request, predictor=predictor)
    await _scan_queue.put(job)
    logging.info(f"📥 [Queue] Job {job_id} for scanId {request.scanId} added. Queue size: {_scan_queue.qsize()}")


async def _send_webhook(webhook_url: str, payload: dict, max_retries: int = 3, auth_token: str = None):
    """Sends the webhook with exponential backoff retries."""
    headers = {
        "ngrok-skip-browser-warning": "true",
        "Content-Type": "application/json"
    }
    if auth_token:
        headers["Authorization"] = auth_token

    try:
        body_str = json.dumps(payload, default=str)
    except Exception as e:
        body_str = f"<JSON SERIALIZATION FAILED: {e}>"
    logging.info(f"📋 [Webhook] Full JSON body being sent:\n{body_str[:2000]}")

    async with httpx.AsyncClient() as client:
        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"🚀 [Webhook] Attempt {attempt} sending to {webhook_url}")
                response = await client.post(webhook_url, content=body_str.encode('utf-8'), headers=headers, timeout=120.0)
                if response.is_error:
                    logging.error(f"❌ [Webhook] Response Status {response.status_code} Body: {response.text}")
                response.raise_for_status()
                logging.info(f"✅ [Webhook] Successfully delivered to {webhook_url}")
                return
            except Exception as e:
                logging.warning(f"⚠️ [Webhook] Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 ** attempt)  # 2s, 4s
                else:
                    logging.error(f"❌ [Webhook] All {max_retries} attempts failed for {webhook_url}.")


async def _worker_loop():
    """Background worker loop that consumes jobs from the queue."""
    logging.info("🛠️ [QueueWorker] Started background worker loop.")
    while True:
        try:
            job: ScanJob = await _scan_queue.get()
        except asyncio.CancelledError:
            logging.info("🛑 [QueueWorker] Worker loop cancelled.")
            break

        try:
            scan_id = job.request.scanId
            req_fields = job.request.model_dump()
            logging.info(
                f"⚙️ [QueueWorker] Processing job {job.job_id} (scanId: '{scan_id}') with request fields:\n" +
                "\n".join(f"  • {k}: {v}" for k, v in req_fields.items())
            )

            try:
                # Run the actual inference logic
                result: AdvancedScanResponseV5 = await run_full_scan_logic(job.request, job.predictor)

                # Build success payload
                payload = WebhookPayload(
                    scanId=scan_id,
                    scanScore=result.scanScore,
                    notes=result.notes,
                    quality=result.quality,
                    mmpose=result.mmpose,
                    yolo=result.yolo
                )

            except Exception as e:
                logging.error(f"❌ [QueueWorker] Job {job.job_id} failed: {e}", exc_info=True)
                # Build failure payload
                payload = WebhookPayload(
                    scanId=scan_id,
                    error=str(e)
                )

            # Send Webhook — convert to dict for httpx
            payload_dict = payload.model_dump(exclude_none=True)
            logging.info(f"📤 [QueueWorker] Delivering payload for scanId='{scan_id}' with keys: {list(payload_dict.keys())}")

            # Read static webhook base URL and Auth Token from .env
            load_dotenv(override=True)
            node_base_url = os.getenv("NODE_BASE_URL")
            auth_token = os.getenv("WEBHOOK_AUTH_TOKEN")

            if node_base_url:
                webhook_url = f"{node_base_url.rstrip('/')}/v1/webhook/horse-scan"
                await _send_webhook(webhook_url, payload_dict, auth_token=auth_token)
            else:
                logging.error("❌ [QueueWorker] NODE_BASE_URL is not set in .env! Cannot send results.")

        except Exception as e:
            logging.error(f"❌ [QueueWorker] Unexpected error in worker loop: {e}", exc_info=True)
        finally:
            _scan_queue.task_done()


def start_queue_worker():
    """Starts the background worker task."""
    global _worker_task
    if _worker_task is None:
        _worker_task = asyncio.create_task(_worker_loop())


async def stop_queue_worker():
    """Stops the background worker task gracefully."""
    global _worker_task
    if _worker_task:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
        _worker_task = None
        logging.info("🛑 [QueueWorker] Stopped gracefully.")

