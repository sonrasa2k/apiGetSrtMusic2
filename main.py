import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from faster_whisper import WhisperModel
from audio_separator.separator import Separator


app = FastAPI(
    title="MP3 to SRT API",
    description="API chuyển đổi file MP3 thành file SRT sử dụng Faster-Whisper",
    version="1.0.0"
)

# Load Whisper model
model = WhisperModel("medium", device="cpu", compute_type="int8")

# Load Audio Separator
separator = Separator()
separator.load_model("UVR-MDX-NET-Voc_FT.onnx")


def separate_vocals(audio_path: str, output_dir: str) -> str:
    """Tách vocals từ audio file, trả về path đến file vocals"""
    # Cấu hình output directory
    separator.output_dir = output_dir

    # Tách vocals
    output_files = separator.separate(audio_path)

    # Tìm file vocals trong kết quả
    for f in output_files:
        if "Vocals" in f or "vocal" in f.lower():
            return f

    # Nếu có output, dùng file đầu tiên
    if output_files:
        return output_files[0]

    # Nếu không tìm thấy, dùng file gốc
    return audio_path


def format_timestamp(seconds: float) -> str:
    """Chuyển đổi giây thành định dạng SRT timestamp (HH:MM:SS,mmm)"""
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
        srt_lines.append("")  # Dòng trống giữa các subtitle

    return "\n".join(srt_lines)


@app.post("/transcribe", response_class=PlainTextResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="File MP3 cần chuyển đổi"),
    language: str = None
):
    """
    Chuyển đổi file MP3 thành nội dung SRT.

    - **file**: File MP3 cần transcribe
    - **language**: Mã ngôn ngữ (vd: vi, en, ja). Để trống để tự động nhận diện.

    Trả về nội dung file SRT dưới dạng text.
    """
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

    try:
        # Ghi file upload vào file tạm
        with open(temp_audio_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Tách vocals trước khi transcribe
        vocals_path = separate_vocals(temp_audio_path, separated_dir)

        # Transcribe với faster-whisper
        segments, info = model.transcribe(
            vocals_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            hallucination_silence_threshold=0.5
        )

        # Chuyển generator thành list để xử lý
        segments_list = list(segments)

        if not segments_list:
            raise HTTPException(status_code=400, detail="Không thể nhận diện nội dung audio")

        # Tạo nội dung SRT
        srt_content = create_srt_content(segments_list)

        return PlainTextResponse(
            content=srt_content,
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{os.path.splitext(file.filename)[0]}.srt"'
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")

    finally:
        # Dọn dẹp file tạm
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/transcribe/download")
async def transcribe_and_download(
    file: UploadFile = File(..., description="File MP3 cần chuyển đổi"),
    language: str = None
):
    """
    Chuyển đổi file MP3 và download file SRT.

    - **file**: File MP3 cần transcribe
    - **language**: Mã ngôn ngữ (vd: vi, en, ja). Để trống để tự động nhận diện.

    Trả về file SRT để download.
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.flac', '.ogg')):
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ các định dạng: mp3, wav, m4a, flac, ogg"
        )

    temp_dir = tempfile.mkdtemp()
    temp_audio_path = os.path.join(temp_dir, file.filename)
    separated_dir = os.path.join(temp_dir, "separated")
    os.makedirs(separated_dir, exist_ok=True)
    srt_filename = os.path.splitext(file.filename)[0] + ".srt"
    temp_srt_path = os.path.join(temp_dir, srt_filename)

    try:
        with open(temp_audio_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Tách vocals trước khi transcribe
        vocals_path = separate_vocals(temp_audio_path, separated_dir)

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
            raise HTTPException(status_code=400, detail="Không thể nhận diện nội dung audio")

        srt_content = create_srt_content(segments_list)

        # Ghi file SRT
        with open(temp_srt_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        return FileResponse(
            path=temp_srt_path,
            filename=srt_filename,
            media_type="application/x-subrip",
            background=None
        )

    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@app.get("/")
async def root():
    """API Info"""
    return {
        "message": "MP3 to SRT API",
        "docs": "/docs",
        "endpoints": {
            "/transcribe": "POST - Chuyển MP3 thành SRT (trả về text)",
            "/transcribe/download": "POST - Chuyển MP3 thành SRT (download file)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
