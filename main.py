from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz
from PyPDF2 import PdfReader
import pdfplumber
from docx import Document
from pdf2image import convert_from_bytes
import pytesseract
import tempfile
import os
import io
import traceback
from PIL import ImageEnhance
import logging
from openai import AsyncOpenAI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CV Analysis API",
    description="AI-Powered CV Analysis and Feedback",
    version="1.0.0"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize Redis
redis_client = redis.Redis(
    host=os.getenv("RENDER_REDIS_HOST", "localhost"),
    port=int(os.getenv("RENDER_REDIS_PORT", 6379)),
    password=os.getenv("RENDER_REDIS_PASSWORD", ""),
    decode_responses=True
)

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Add CORS middleware
origins = [
    "https://frontend-portfolio-aomn.onrender.com",
    "https://deerk-portfolio.onrender.com",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:4173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@app.options("/{path:path}")
async def options_route(request: Request):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600",
        }
    )

from ai_analyzer import AIAnalyzer

async def extract_text_from_pdf_with_ocr(pdf_content: bytes) -> str:
    """Extract text from PDF using OCR as a last resort."""
    try:
        # First try PyMuPDF (fitz)
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        if text.strip():
            return text

        # If no text found, try pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            
            if text.strip():
                return text

        # If still no text, use OCR
        images = convert_from_bytes(pdf_content)
        text = ""
        for image in images:
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(image)
            enhanced_image = enhancer.enhance(2.0)
            text += pytesseract.image_to_string(enhanced_image)

        return text.strip() or "No text could be extracted from the PDF"

    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

@app.get("/")
@limiter.limit("10/minute")
def root(request: Request):
    """Health check endpoint."""
    return {"status": "ok", "message": "CV Analysis Service is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/supported-jobs")
async def get_supported_jobs():
    """Get list of supported job titles for targeted analysis."""
    try:
        ai_analyzer = AIAnalyzer()
        return {"supported_jobs": list(ai_analyzer.industry_requirements.keys())}
    except Exception as e:
        logging.error(f"Error getting supported jobs: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error getting supported jobs: {str(e)}"
        )

@app.post("/analyze")
@limiter.limit("10/minute")
async def analyze_cv(
    request: Request,
    cv_file: UploadFile = File(...),
    job_category: str = Form(...),
    include_industry_insights: bool = Form(False),
    include_competitive_analysis: bool = Form(False),
    detailed_feedback: bool = Form(False)
):
    try:
        # Read and process the CV file
        content = await cv_file.read()
        
        # Extract text based on file type
        if cv_file.filename.lower().endswith('.pdf'):
            text = await extract_text_from_pdf_with_ocr(content)
        elif cv_file.filename.lower().endswith('.docx'):
            doc = Document(io.BytesIO(content))
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF or DOCX file.")

        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the file")

        # Initialize AI analyzer
        analyzer = AIAnalyzer()
        
        # Perform analysis
        try:
            analysis_result = await analyzer.analyze_resume(
                text=text,
                job_title=job_category,
                include_industry_insights=include_industry_insights,
                include_competitive_analysis=include_competitive_analysis,
                detailed_feedback=detailed_feedback
            )
        except Exception as e:
            logging.error(f"Error during CV analysis: {str(e)}")
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing CV: {str(e)}"
            )

        return {
            "status": "success",
            "analysis": analysis_result
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Unexpected error in analyze_cv: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        workers=4
    )
