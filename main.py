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

app = FastAPI()

# Initialize OpenAI client
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Add CORS middleware
origins = [
    "https://frontend-portfolio-aomn.onrender.com",
    "http://localhost:3000",  # For local development
    "http://localhost:5173"   # For Vite dev server
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{path:path}")
async def options_route(request: Request):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

from ai_analyzer import AIAnalyzer

def extract_text_from_pdf_with_ocr(pdf_content):
    """Extract text from PDF using OCR as a last resort."""
    try:
        # Convert PDF to images with higher DPI for better quality
        images = convert_from_bytes(pdf_content, dpi=400)
        text = ""
        
        for image in images:
            # Enhanced image preprocessing for better OCR
            img = image.convert('L')  # Convert to grayscale
            # Increase contrast
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            
            # Use advanced OCR configuration
            custom_config = r'--oem 3 --psm 6 -l eng --dpi 400'
            page_text = pytesseract.image_to_string(img, config=custom_config)
            
            # Basic text cleanup
            page_text = page_text.replace('\x0c', '\n').strip()
            if page_text:
                text += page_text + "\n\n"
        
        return text.strip()
    except Exception as e:
        print(f"OCR processing error: {str(e)}")
        traceback.print_exc()
        raise Exception(f"OCR error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "CV Critique API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/supported-jobs")
async def get_supported_jobs():
    """Get list of supported job titles for targeted analysis."""
    ai_analyzer = AIAnalyzer()
    return {"supported_jobs": list(ai_analyzer.industry_requirements.keys())}

@app.post("/analyze")
async def analyze_cv(
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
            text = extract_text_from_pdf_with_ocr(content)
        elif cv_file.filename.lower().endswith('.docx'):
            doc = Document(io.BytesIO(content))
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a PDF or DOCX file.")

        # Initialize AI analyzer
        analyzer = AIAnalyzer()
        
        # Perform analysis
        analysis_result = await analyzer.analyze_resume(
            text=text,
            job_title=job_category,
            include_industry_insights=include_industry_insights,
            include_competitive_analysis=include_competitive_analysis,
            detailed_feedback=detailed_feedback
        )

        return {
            "status": "success",
            "analysis": analysis_result
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable
    port = int(os.getenv("PORT", 10000))
    
    # Add debug logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.debug(f"Starting server on port {port}")
    
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="debug")
