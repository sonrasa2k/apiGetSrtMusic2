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


# Queue và worker config
MAX_WORKERS = 2
task_queue: asyncio.Queue = None
executor: ThreadPoolExecutor = None

# Task storage
tasks: dict = {}

# Models
model: WhisperModel = None
separator: Separator = None


def load_models():
    """Load models một lần khi startup"""
    global model, separator
    print("[INFO] Loading Whisper model...")
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    print("[INFO] Loading Audio Separator model...")
    separator = Separator()
    separator.load_model("UVR-MDX-NET-Voc_FT.onnx")
    print("[INFO] Models loaded!")


def separate_vocals(audio_path: str, output_dir: str) -> str:
    """Tách vocals từ audio file"""
    separator.output_dir = output_dir
    output_files = separator.separate(audio_path)

    for f in output_files:
        if "Vocals" in f or "vocal" in f.lower():
            return f

    if output_files:
        return output_files[0]

    return audio_path


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_content(segments) -> str:
    srt_lines = []
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment.start)
        end_time = format_timestamp(segment.end)
        text = segment.text.strip()
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start_time} --> {end_time}")
        srt_lines.append(text)
        srt_lines.append("")
    return "\n".join(srt_lines)


def process_audio_task(task_id: str, task_data: dict):
    """Xử lý task trong thread pool"""
    audio_path = task_data["audio_path"]
    separated_dir = task_data["separated_dir"]
    language = task_data["language"]
    use_separator = task_data["use_separator"]

    try:
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["started_at"] = time.time()

        if use_separator:
            tasks[task_id]["step"] = "separating"
            vocals_path = separate_vocals(audio_path, separated_dir)
        else:
            vocals_path = audio_path

        tasks[task_id]["step"] = "transcribing"

        segments, info = model.transcribe(
            vocals_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=0.5
        )

        segments_list = list(segments)
        if not segments_list:
            raise Exception("Không thể nhận diện nội dung audio")

        srt_content = create_srt_content(segments_list)

        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = srt_content
        tasks[task_id]["completed_at"] = time.time()

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["completed_at"] = time.time()

    finally:
        temp_dir = task_data.get("temp_dir")
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


async def worker(worker_id: int):
    """Worker xử lý task từ queue"""
    print(f"[INFO] Worker {worker_id} started")
    while True:
        task_id, task_data = await task_queue.get()
        print(f"[INFO] Worker {worker_id} processing task {task_id}")

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, process_audio_task, task_id, task_data)
        except Exception as e:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
        finally:
            task_queue.task_done()
            print(f"[INFO] Worker {worker_id} finished task {task_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global task_queue, executor

    load_models()
    task_queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    workers = [asyncio.create_task(worker(i)) for i in range(MAX_WORKERS)]
    print(f"[INFO] Started {MAX_WORKERS} workers")

    yield

    for w in workers:
        w.cancel()
    executor.shutdown(wait=True)


app = FastAPI(
    title="MP3 to SRT API",
    description="API chuyển đổi MP3 thành SRT với hàng đợi async",
    version="3.0.0",
    lifespan=lifespan
)


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(..., description="File MP3 cần chuyển đổi"),
    language: str = None
):
    """
    Submit task transcribe (có tách vocals).
    Trả về task_id để check status sau.
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ: mp3, wav, m4a, flac, ogg")

    # Lưu file tạm
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, file.filename)
    separated_dir = os.path.join(temp_dir, "separated")
    os.makedirs(separated_dir, exist_ok=True)

    with open(temp_audio_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Tạo task
    task_id = str(uuid.uuid4())[:8]
    task_data = {
        "audio_path": temp_audio_path,
        "separated_dir": separated_dir,
        "language": language,
        "use_separator": True,
        "temp_dir": temp_dir
    }

    # Lưu task info
    queue_position = task_queue.qsize() + 1
    tasks[task_id] = {
        "id": task_id,
        "filename": file.filename,
        "status": "queued",
        "queue_position": queue_position,
        "step": "waiting",
        "created_at": time.time(),
        "result": None,
        "error": None
    }

    # Thêm vào queue
    await task_queue.put((task_id, task_data))
    print(f"[INFO] Task {task_id} added to queue (position: {queue_position})")

    return {
        "task_id": task_id,
        "status": "queued",
        "queue_position": queue_position,
        "message": f"Task đã được thêm vào hàng đợi. Dùng GET /task/{task_id} để check status."
    }


@app.post("/transcribe-simple")
async def transcribe_simple(
    file: UploadFile = File(...),
    language: str = None
):
    """Submit task transcribe (không tách vocals)"""
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ: mp3, wav, m4a, flac, ogg")

    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, file.filename)

    with open(temp_audio_path, "wb") as f:
        content = await file.read()
        f.write(content)

    task_id = str(uuid.uuid4())[:8]
    task_data = {
        "audio_path": temp_audio_path,
        "separated_dir": temp_dir,
        "language": language,
        "use_separator": False,
        "temp_dir": temp_dir
    }

    queue_position = task_queue.qsize() + 1
    tasks[task_id] = {
        "id": task_id,
        "filename": file.filename,
        "status": "queued",
        "queue_position": queue_position,
        "step": "waiting",
        "created_at": time.time(),
        "result": None,
        "error": None
    }

    await task_queue.put((task_id, task_data))

    return {
        "task_id": task_id,
        "status": "queued",
        "queue_position": queue_position
    }


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Check status của task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task không tồn tại")

    task = tasks[task_id]
    response = {
        "task_id": task_id,
        "filename": task["filename"],
        "status": task["status"],
        "step": task["step"]
    }

    if task["status"] == "queued":
        response["queue_position"] = task["queue_position"]

    if task["status"] == "processing":
        elapsed = time.time() - task.get("started_at", task["created_at"])
        response["elapsed_seconds"] = round(elapsed, 1)

    if task["status"] == "completed":
        elapsed = task["completed_at"] - task["created_at"]
        response["total_seconds"] = round(elapsed, 1)
        response["message"] = f"Hoàn thành! Dùng GET /task/{task_id}/result để lấy kết quả."

    if task["status"] == "failed":
        response["error"] = task["error"]

    return response


@app.get("/task/{task_id}/result", response_class=PlainTextResponse)
async def get_task_result(task_id: str):
    """Lấy kết quả SRT của task đã hoàn thành"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task không tồn tại")

    task = tasks[task_id]

    if task["status"] == "queued":
        raise HTTPException(status_code=202, detail="Task đang trong hàng đợi")

    if task["status"] == "processing":
        raise HTTPException(status_code=202, detail="Task đang xử lý")

    if task["status"] == "failed":
        raise HTTPException(status_code=500, detail=task["error"])

    if task["status"] == "completed":
        # Xóa task sau khi lấy kết quả
        result = task["result"]
        del tasks[task_id]
        return PlainTextResponse(
            content=result,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.splitext(task["filename"])[0]}.srt"'
            }
        )


@app.get("/tasks")
async def list_tasks():
    """Liệt kê tất cả tasks"""
    task_list = []
    for task_id, task in tasks.items():
        task_list.append({
            "task_id": task_id,
            "filename": task["filename"],
            "status": task["status"],
            "step": task["step"]
        })
    return {
        "total": len(task_list),
        "tasks": task_list
    }


@app.get("/queue-status")
async def queue_status():
    """Xem trạng thái hàng đợi"""
    return {
        "workers": MAX_WORKERS,
        "queue_size": task_queue.qsize() if task_queue else 0,
        "total_tasks": len(tasks)
    }


@app.get("/")
async def root():
    queue_size = task_queue.qsize() if task_queue else 0
    return {
        "message": "MP3 to SRT API",
        "version": "3.0.0",
        "workers": MAX_WORKERS,
        "queue_size": queue_size,
        "docs": "/docs",
        "endpoints": {
            "POST /transcribe": "Submit task (có tách vocals) → trả task_id",
            "POST /transcribe-simple": "Submit task (không tách vocals) → trả task_id",
            "GET /task/{id}": "Check status của task",
            "GET /task/{id}/result": "Lấy kết quả SRT",
            "GET /tasks": "Liệt kê tất cả tasks",
            "GET /queue-status": "Xem trạng thái queue"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
