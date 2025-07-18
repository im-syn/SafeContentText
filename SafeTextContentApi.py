#!/usr/bin/env python3
"""
SafeTextContentApi.py

Web API endpoint for bad-content detection in text using a zero-shot AI classifier.
Provides configurable GET/POST /detect endpoint (supports both 'texts' list and
optional single 'text' param), and POST /detect/file for file uploads.
Returns JSON with per-text scores, flagged labels, and optional is_safe flag.
"""

import os
import logging
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Form, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import pipeline, Pipeline

# --- Configurable options ---
ALLOW_GET: bool = True                # Enable GET /detect
ALLOW_POST: bool = True               # Enable POST /detect
ALLOW_FILE_UPLOAD: bool = True        # Enable POST /detect/file
ENABLE_CORS: bool = True              # Enable CORS middleware
ENABLE_TEXT_PARAM: bool = True        # Allow 'text' (singular) query param
ENABLE_FAST_DETECT: bool = True       # Add 'is_safe' flag in results
HOST: str = os.getenv("STC_API_HOST", "127.0.0.1")
PORT: int = int(os.getenv("STC_API_PORT", 8989))
RELOAD: bool = bool(os.getenv("STC_API_RELOAD", True))
CACHE_DIR: str = os.getenv("HF_CACHE_DIR", os.path.join(os.getcwd(), "hf_cache"))

# Default labels to spot
DEFAULT_LABELS: List[str] = [
    "profanity",
    "hate speech",
    "graphic violence",
    "self-harm",
    "sexual content",
    "insult",
    "terrorism"
]

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Load zero-shot classifier once at startup
logging.info(f"Loading zero-shot classifier into cache at {CACHE_DIR}...")
classifier: Pipeline = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    cache_dir=CACHE_DIR
)

# --- Pydantic models for request & response ---

class DetectRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        description="List of text strings to analyze",
        example=["This is some hateful gore content!"]
    )
    labels: Optional[List[str]] = Field(
        DEFAULT_LABELS,
        description="Which categories to detect",
        example=DEFAULT_LABELS
    )
    threshold: Optional[float] = Field(
        0.5,
        ge=0.0, le=1.0,
        description="Score threshold above which a label is flagged",
        example=0.5
    )

class DetectResult(BaseModel):
    text: str
    scores: Dict[str, float]
    flagged_labels: Dict[str, float]
    is_safe: Optional[bool] = None

class DetectResponse(BaseModel):
    results: List[DetectResult]

# --- FastAPI app setup ---

app = FastAPI(
    title="SafeTextContent API",
    description="API for detecting profanity, hate, gore, etc., in text.",
    version="1.0.0"
)

# Conditional CORS
if ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Classification helper ---

def classify_texts(
    texts: List[str],
    labels: List[str],
    threshold: float
) -> List[DetectResult]:
    """
    Run zero-shot classification and return list of DetectResult.
    """
    try:
        raw = classifier(texts, candidate_labels=labels, multi_label=True)
    except Exception as e:
        logging.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail="Internal classification error")

    if isinstance(raw, dict):
        raw = [raw]

    results: List[DetectResult] = []
    for text, res in zip(texts, raw):
        label_scores = dict(zip(res["labels"], res["scores"]))
        flagged = {lbl: sc for lbl, sc in label_scores.items() if sc >= threshold}
        is_safe = (len(flagged) == 0) if ENABLE_FAST_DETECT else None
        results.append(
            DetectResult(
                text=text,
                scores=label_scores,
                flagged_labels=flagged,
                is_safe=is_safe
            )
        )
    return results

# --- Error handlers ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": "Validation Error", "details": exc.errors()}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error"}
    )

# --- Endpoint definitions ---

if ALLOW_POST:
    @app.post(
        "/detect",
        response_model=DetectResponse,
        response_model_exclude_none=True
    )
    async def detect_post(request: DetectRequest) -> DetectResponse:
        """
        POST JSON {texts, labels?, threshold?}
        """
        labels = request.labels or DEFAULT_LABELS
        threshold = request.threshold if request.threshold is not None else 0.5
        return DetectResponse(results=classify_texts(request.texts, labels, threshold))

if ALLOW_GET:
    @app.get(
        "/detect",
        response_model=DetectResponse,
        response_model_exclude_none=True
    )
    async def detect_get(
        texts: Optional[List[str]] = Query(
            None,
            description="List of text strings to analyze; use multiple texts=param"
        ),
        text: Optional[str] = Query(
            None,
            description="Single text string to analyze (requires ENABLE_TEXT_PARAM)"
        ),
        labels: Optional[List[str]] = Query(
            DEFAULT_LABELS,
            description="Which categories to detect"
        ),
        threshold: float = Query(
            0.5, ge=0.0, le=1.0,
            description="Score threshold above which a label is flagged"
        )
    ) -> DetectResponse:
        """
        GET /detect?texts=one&texts=two or /detect?text=single
        """
        if text is not None:
            if not ENABLE_TEXT_PARAM:
                raise HTTPException(
                    status_code=400,
                    detail="Single-text query param 'text' not supported"
                )
            texts_list = [text]
        elif texts:
            texts_list = texts
        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either 'texts' or 'text' query parameter"
            )
        return DetectResponse(
            results=classify_texts(
                texts_list,
                labels or DEFAULT_LABELS,
                threshold
            )
        )

if ALLOW_FILE_UPLOAD:
    @app.post(
        "/detect/file",
        response_model=DetectResponse,
        response_model_exclude_none=True
    )
    async def detect_file(
        file: UploadFile = File(..., description="Text file to analyze (.txt)"),
        labels: List[str] = Form(
            DEFAULT_LABELS,
            description="Which categories to detect"
        ),
        threshold: float = Form(
            0.5,
            ge=0.0, le=1.0,
            description="Score threshold above which a label is flagged"
        )
    ) -> DetectResponse:
        """
        POST file upload. file must be text/plain (.txt).
        Lines → separate texts.
        """
        if file.content_type != "text/plain":
            raise HTTPException(status_code=400, detail="Only text/plain supported.")
        content = (await file.read()).decode("utf-8")
        texts_list = content.splitlines() if "\n" in content else [content]
        return DetectResponse(results=classify_texts(texts_list, labels, threshold))

# --- Uvicorn runner ---
# run: uvicorn SafeTextContentApi:app --host localhost --port 8989 --reload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "SafeTextContentApi:app",
        host=HOST,
        port=PORT,
        reload=RELOAD
    )
