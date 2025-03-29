import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from process_query import chain, FinancialState  # Assuming this module contains the LLM pipeline
from typing import List, Dict, Any
from fastapi.responses import FileResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from utils import generate_audio_summary
import requests


DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if not os.path.exists("reports"):
    os.makedirs("reports")

# Mount the static directory to serve audio files
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

if not os.path.exists("static"):
    os.makedirs("static")

# Mount the static directory to serve audio files
app.mount("/static", StaticFiles(directory="static"), name="static")



class QueryRequest(BaseModel):
    user_query: str

class FinancialResponse(BaseModel):
    symbol: str
    summary: str
    report_link: str
    transcription: str
    anomalies_detected: List[int]
    error: str | None = None

REPORTS_DIR = Path("./reports")  # Directory where reports are saved



@app.post("/process_query", response_model=FinancialResponse)
async def process_query(request: QueryRequest):
    try:
        result = chain.invoke(FinancialState(user_query=request.user_query))
        result = FinancialState(**result)

        if result.error:
            raise HTTPException(status_code=400, detail=result.error)

         # transcription = generate_audio_summary(result.insights)
        return {
            "symbol": result.symbol,
            "summary": result.insights,
            "report_link": result.report_link,
            "transcription" : "",
            "anomalies_detected": result.anomalies
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System error: {str(e)}")

@app.get("/reports/{filename}")
async def get_report(filename: str):
    """
    Fetch a generated stock analysis report by filename.
    Example: /reports/NVDA_report.html
    """
    report_path = REPORTS_DIR / filename
    if report_path.exists():
        return FileResponse(report_path, media_type="text/html")
    return {"error": "Report not found"}

@app.post("/transcribe_audio")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Receives an audio file and sends it to Deepgram for transcription.
    Returns the transcribed text.
    """

    # Read the uploaded file into memory
    audio_bytes = await file.read()

    # Deepgram API URL
    url = "https://api.deepgram.com/v1/listen"

    # Headers for Deepgram API
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": file.content_type,  # Automatically detects file type
    }

    # Send Audio to Deepgram
    response = requests.post(url, headers=headers, data=audio_bytes)

    # Check for errors
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"Deepgram Error: {response.json()}")

    # Extract Transcription
    transcript_data = response.json()
    transcript_text = transcript_data.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")

    return {"transcription": transcript_text}
