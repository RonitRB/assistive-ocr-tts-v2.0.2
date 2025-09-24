# app.py
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging
import os

from core.config import Config
from core.pipeline import AssistivePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("assistive_app")

app = FastAPI(title="Assistive OCR→TTS")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

cfg = Config(os.path.join(BASE_DIR, "config.json"))
pipeline = AssistivePipeline(cfg)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    voices = []
    if pipeline.tts.coqui and hasattr(pipeline.tts.coqui, 'speakers') and pipeline.tts.coqui.speakers:
        voices = pipeline.tts.coqui.speakers
    return templates.TemplateResponse("dashboard.html", {"request": request, "config": cfg.data, "voices": voices})

@app.post("/api/start")
async def api_start():
    pipeline.start()
    return JSONResponse({"status": "started", "pipeline": pipeline.get_status()})

@app.post("/api/stop")
async def api_stop():
    pipeline.stop()
    return JSONResponse({"status": "stopped", "pipeline": pipeline.get_status()})

@app.get("/api/status")
async def api_status():
    return JSONResponse({"status": "ok", "pipeline": pipeline.get_status()})

@app.get("/api/history")
async def api_history():
    return JSONResponse({"history": pipeline.get_history()})

@app.get("/api/config")
async def api_get_config():
    return JSONResponse(cfg.data)

@app.post("/api/config")
async def api_update_config(payload: dict):
    global pipeline
    cfg.update(payload)
    pipeline.stop()
    pipeline = AssistivePipeline(cfg)
    pipeline.start()
    return JSONResponse({"status": "saved", "config": cfg.data})


@app.post("/api/speak")
async def api_speak(payload: dict):
    text = payload.get("text", "")
    if text:
        pipeline.tts.speak(text,
                          voice=cfg.data["tts"].get("voice"),
                          speed=cfg.data["tts"].get("speed",1.0),
                          volume=cfg.data["tts"].get("volume",0.9))
        return JSONResponse({"status": "speaking"})
    return JSONResponse({"status": "error", "message": "no text"}, status_code=400)

@app.get("/api/replay")
async def api_replay():
    # return last audio file if available
    last = pipeline.tts.last_audio_path
    if last and os.path.exists(last):
        return FileResponse(last, media_type="audio/wav", filename="last_audio.wav")
    return JSONResponse({"status":"no_audio"}, status_code=404)

if __name__ == "__main__":
    # Create templates directory if missing (templates provided separately)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)