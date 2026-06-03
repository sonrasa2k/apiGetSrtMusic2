import os
import tempfile
import shutil
import asyncio
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from faster_whisper import WhisperModel
from audio_separator.separator import Separator


MAX_WORKERS = 2
task_queue: asyncio.Queue = None
executor: ThreadPoolExecutor = None
tasks: dict = {}
model: WhisperModel = None
separator: Separator = None


def load_models():
    global model, separator
    print("[INFO] Loading Whisper model...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print("[INFO] Loading Audio Separator model...")
    separator = Separator()
    separator.load_model("UVR-MDX-NET-Voc_FT.onnx")
    print("[INFO] Models loaded!")


def separate_vocals(audio_path: str, output_dir: str) -> str:
    separator.output_dir = output_dir
    output_files = separator.separate(audio_path)
    for f in output_files:
        if "Vocals" in f or "vocal" in f.lower():
            return f
    return output_files[0] if output_files else audio_path


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_content(segments) -> str:
    srt_lines = []
    for i, segment in enumerate(segments, start=1):
        srt_lines.append(f"{i}")
        srt_lines.append(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}")
        srt_lines.append(segment.text.strip())
        srt_lines.append("")
    return "\n".join(srt_lines)


def process_audio_task(task_id: str, task_data: dict):
    audio_path = task_data["audio_path"]
    language = task_data["language"]
    use_separator = task_data["use_separator"]
    temp_dir = task_data["temp_dir"]

    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["started_at"] = time.time()

        if use_separator:
            tasks[task_id]["step"] = "separating"
            separated_dir = os.path.join(temp_dir, "separated")
            os.makedirs(separated_dir, exist_ok=True)
            audio_path = separate_vocals(audio_path, separated_dir)

        tasks[task_id]["step"] = "transcribing"
        segments, _ = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=0.5
        )

        segments_list = list(segments)
        if not segments_list:
            raise Exception("Không nhận diện được nội dung audio")

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = create_srt_content(segments_list)
        tasks[task_id]["completed_at"] = time.time()

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = time.time()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def worker(worker_id: int):
    print(f"[INFO] Worker {worker_id} started")
    while True:
        task_id, task_data = await task_queue.get()
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, process_audio_task, task_id, task_data)
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
        finally:
            task_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global task_queue, executor
    load_models()
    task_queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    workers = [asyncio.create_task(worker(i)) for i in range(MAX_WORKERS)]
    yield
    for w in workers:
        w.cancel()
    executor.shutdown(wait=True)


app = FastAPI(
    title="Audio to SRT API",
    description="Chuyển audio tiếng Trung thành SRT, hàng đợi async",
    version="4.0.0",
    lifespan=lifespan
)

ALLOWED_EXTS = ('.mp3', '.wav', '.m4a', '.flac', '.ogg')


def _enqueue(file_obj, language: str, use_separator: bool):
    task_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, file_obj["filename"])
    with open(audio_path, "wb") as f:
        f.write(file_obj["content"])
    queue_position = task_queue.qsize() + 1
    tasks[task_id] = {
        "id": task_id,
        "filename": file_obj["filename"],
        "status": "queued",
        "queue_position": queue_position,
        "step": "waiting",
        "created_at": time.time(),
        "result": None,
        "error": None
    }
    return task_id, {"audio_path": audio_path, "language": language,
                     "use_separator": use_separator, "temp_dir": temp_dir}, queue_position


@app.post("/transcribe", status_code=202)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "zh"
):
    """Tách vocals khỏi nhạc nền rồi transcribe → SRT. Dùng cho phim có nhạc nền."""
    if not file.filename.lower().endswith(ALLOWED_EXTS):
        raise HTTPException(400, f"Chỉ hỗ trợ: {', '.join(ALLOWED_EXTS)}")
    file_obj = {"filename": file.filename, "content": await file.read()}
    task_id, task_data, pos = _enqueue(file_obj, language, use_separator=True)
    await task_queue.put((task_id, task_data))
    return {"task_id": task_id, "status": "queued", "queue_position": pos,
            "message": f"GET /task/{task_id} để check status."}


@app.post("/transcribe-simple", status_code=202)
async def transcribe_simple(
    file: UploadFile = File(...),
    language: str = "zh"
):
    """Transcribe thẳng, không tách vocals. Dùng cho audio sạch (podcast, phỏng vấn)."""
    if not file.filename.lower().endswith(ALLOWED_EXTS):
        raise HTTPException(400, f"Chỉ hỗ trợ: {', '.join(ALLOWED_EXTS)}")
    file_obj = {"filename": file.filename, "content": await file.read()}
    task_id, task_data, pos = _enqueue(file_obj, language, use_separator=False)
    await task_queue.put((task_id, task_data))
    return {"task_id": task_id, "status": "queued", "queue_position": pos}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task không tồn tại")
    task = tasks[task_id]
    resp = {"task_id": task_id, "filename": task["filename"],
            "status": task["status"], "step": task["step"]}
    if task["status"] == "queued":
        resp["queue_position"] = task["queue_position"]
    if task["status"] == "processing":
        resp["elapsed_seconds"] = round(time.time() - task.get("started_at", task["created_at"]), 1)
    if task["status"] == "completed":
        resp["total_seconds"] = round(task["completed_at"] - task["created_at"], 1)
        resp["message"] = f"Xong! GET /task/{task_id}/result để tải SRT."
    if task["status"] == "failed":
        resp["error"] = task["error"]
    return resp


@app.get("/task/{task_id}/result", response_class=PlainTextResponse)
async def get_task_result(task_id: str):
    if task_id not in tasks:
        raise HTTPException(404, "Task không tồn tại")
    task = tasks[task_id]
    if task["status"] == "queued":
        raise HTTPException(202, "Đang trong hàng đợi")
    if task["status"] == "processing":
        raise HTTPException(202, "Đang xử lý")
    if task["status"] == "failed":
        raise HTTPException(500, task["error"])
    result = task["result"]
    del tasks[task_id]
    stem = os.path.splitext(task["filename"])[0]
    return PlainTextResponse(
        content=result,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{stem}.srt"'}
    )


@app.get("/tasks")
async def list_tasks():
    return {"total": len(tasks), "tasks": [
        {"task_id": k, "filename": v["filename"], "status": v["status"]}
        for k, v in tasks.items()
    ]}


@app.get("/queue-status")
async def queue_status():
    return {"workers": MAX_WORKERS,
            "queue_size": task_queue.qsize() if task_queue else 0,
            "total_tasks": len(tasks)}


@app.get("/")
async def root():
    return {
        "api": "Audio to SRT (Chinese)",
        "version": "4.0.0",
        "endpoints": {
            "POST /transcribe": "Upload audio → SRT (có tách vocals, dùng cho phim)",
            "POST /transcribe-simple": "Upload audio → SRT (không tách vocals, audio sạch)",
            "GET /task/{id}": "Check status",
            "GET /task/{id}/result": "Tải file SRT",
            "GET /tasks": "Danh sách tasks",
            "GET /queue-status": "Trạng thái queue"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
