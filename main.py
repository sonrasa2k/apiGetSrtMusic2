import os
import tempfile
import shutil
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from faster_whisper import WhisperModel
from audio_separator.separator import Separator


# Queue và worker config
MAX_WORKERS = 2
task_queue: asyncio.Queue = None
executor: ThreadPoolExecutor = None
results: dict = {}

# Models (load một lần, dùng chung)
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
    """Tách vocals từ audio file, trả về path đến file vocals"""
    separator.output_dir = output_dir
    output_files = separator.separate(audio_path)

    print(f"[DEBUG] Output files: {output_files}")

    for f in output_files:
        if "Vocals" in f or "vocal" in f.lower():
            print(f"[DEBUG] Found vocals file: {f}")
            return f

    if output_files:
        print(f"[DEBUG] Using first file: {output_files[0]}")
        return output_files[0]

    print(f"[DEBUG] No output, using original: {audio_path}")
    return audio_path


def format_timestamp(seconds: float) -> str:
    """Chuyển đổi giây thành định dạng SRT timestamp"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt_content(segments) -> str:
    """Tạo nội dung file SRT từ các segments"""
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


def process_audio_task(task_data: dict) -> str:
    """Xử lý task trong thread pool (blocking)"""
    audio_path = task_data["audio_path"]
    separated_dir = task_data["separated_dir"]
    language = task_data["language"]
    use_separator = task_data["use_separator"]

    try:
        if use_separator:
            vocals_path = separate_vocals(audio_path, separated_dir)
        else:
            vocals_path = audio_path

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

        return create_srt_content(segments_list)

    except Exception as e:
        raise Exception(f"Lỗi xử lý: {str(e)}")


async def worker(worker_id: int):
    """Worker xử lý task từ queue"""
    print(f"[INFO] Worker {worker_id} started")
    while True:
        task_id, task_data, event = await task_queue.get()
        print(f"[INFO] Worker {worker_id} processing task {task_id}")

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, process_audio_task, task_data)
            results[task_id] = {"status": "success", "data": result}
        except Exception as e:
            results[task_id] = {"status": "error", "error": str(e)}
        finally:
            # Cleanup temp files
            temp_dir = task_data.get("temp_dir")
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

            event.set()
            task_queue.task_done()
            print(f"[INFO] Worker {worker_id} finished task {task_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup và shutdown"""
    global task_queue, executor

    # Startup
    load_models()
    task_queue = asyncio.Queue()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    # Start workers
    workers = [asyncio.create_task(worker(i)) for i in range(MAX_WORKERS)]
    print(f"[INFO] Started {MAX_WORKERS} workers")

    yield

    # Shutdown
    for w in workers:
        w.cancel()
    executor.shutdown(wait=True)


app = FastAPI(
    title="MP3 to SRT API",
    description="API chuyển đổi file MP3 thành file SRT với hàng đợi xử lý",
    version="2.0.0",
    lifespan=lifespan
)


async def submit_task(file: UploadFile, language: str, use_separator: bool) -> str:
    """Submit task vào queue và chờ kết quả"""
    # Validate file
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ các định dạng: mp3, wav, m4a, flac, ogg"
        )

    # Lưu file tạm
    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, file.filename)
    separated_dir = os.path.join(temp_dir, "separated")
    os.makedirs(separated_dir, exist_ok=True)

    with open(temp_audio_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Tạo task
    task_id = str(uuid.uuid4())
    task_data = {
        "audio_path": temp_audio_path,
        "separated_dir": separated_dir,
        "language": language,
        "use_separator": use_separator,
        "temp_dir": temp_dir
    }

    # Event để chờ kết quả
    event = asyncio.Event()

    # Thêm vào queue
    queue_size = task_queue.qsize()
    print(f"[INFO] Task {task_id} added to queue (position: {queue_size + 1})")

    await task_queue.put((task_id, task_data, event))

    # Chờ worker xử lý xong
    await event.wait()

    # Lấy kết quả
    result = results.pop(task_id)
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["error"])

    return result["data"]


@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="File MP3 cần chuyển đổi"),
    language: str = None
):
    """
    Chuyển đổi file MP3 thành nội dung SRT (có tách vocals).
    Request sẽ được đưa vào hàng đợi và xử lý tuần tự.
    """
    srt_content = await submit_task(file, language, use_separator=True)
    return PlainTextResponse(
        content=srt_content,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{os.path.splitext(file.filename)[0]}.srt"'
        }
    )


@app.post("/transcribe-simple", response_class=PlainTextResponse)
async def transcribe_simple(
    file: UploadFile = File(...),
    language: str = None
):
    """Chuyển MP3 thành SRT (không tách vocals)"""
    srt_content = await submit_task(file, language, use_separator=False)
    return PlainTextResponse(content=srt_content)


@app.get("/")
async def root():
    """API Info"""
    queue_size = task_queue.qsize() if task_queue else 0
    return {
        "message": "MP3 to SRT API",
        "version": "2.0.0",
        "workers": MAX_WORKERS,
        "queue_size": queue_size,
        "docs": "/docs",
        "endpoints": {
            "/transcribe": "POST - Chuyển MP3 thành SRT (có tách vocals)",
            "/transcribe-simple": "POST - Chuyển MP3 thành SRT (không tách vocals)",
            "/queue-status": "GET - Xem trạng thái hàng đợi"
        }
    }


@app.get("/queue-status")
async def queue_status():
    """Xem trạng thái hàng đợi"""
    return {
        "workers": MAX_WORKERS,
        "queue_size": task_queue.qsize() if task_queue else 0,
        "pending_results": len(results)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
