from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app = FastAPI()

# Add CORS middleware with specific frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://frontend-portfolio-aomn.onrender.com",
        "http://localhost:3000",  # For local development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    return {"message": "Resume Critique API is running"}

@app.get("/supported-jobs")
async def get_supported_jobs():
    """Get list of supported job titles for targeted analysis."""
    ai_analyzer = AIAnalyzer()
    return {"supported_jobs": list(ai_analyzer.industry_requirements.keys())}

@app.post("/analyze-resume/")
async def analyze_resume(
    file: UploadFile = File(...),
    job_title: str = Form(None)
):
    try:
        content = await file.read()
        print(f"Processing file: {file.filename}, size: {len(content)} bytes")
        
        # Debug: Check file content
        if len(content) > 0:
            # Check if file starts with PDF signature
            is_pdf_signature = content.startswith(b'%PDF-')
            print(f"File starts with PDF signature: {is_pdf_signature}")
            if not is_pdf_signature and file.filename.lower().endswith('.pdf'):
                print("Warning: File has .pdf extension but doesn't start with PDF signature")
                print(f"First 20 bytes of file: {content[:20]}")
        
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )

        file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
        if file_ext not in ['pdf', 'docx']:
            raise HTTPException(
                status_code=400,
                detail="Only PDF and DOCX files are supported"
            )

        # For PDF files, validate PDF structure
        if file_ext == 'pdf':
            if not content.startswith(b'%PDF-'):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file: File does not start with PDF signature"
                )

        text = None
        error_messages = []
        processing_log = []

        if file_ext == 'docx':
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
                    temp_file.write(content)
                    temp_file.flush()
                    
                    doc = Document(temp_file.name)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    
                    os.unlink(temp_file.name)
                    
                if text.strip():
                    processing_log.append(f"Successfully extracted {len(text)} characters from DOCX")
                else:
                    error_messages.append("DOCX: No text extracted")
            except Exception as e:
                error_messages.append(f"DOCX error: {str(e)}")
                traceback.print_exc()
        else:  # PDF
            # Try PyMuPDF first (usually best for text-based PDFs)
            if not text:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(content)
                        temp_file.flush()
                        
                        pdf_document = fitz.open(temp_file.name)
                        text = ""
                        for page_num in range(pdf_document.page_count):
                            page = pdf_document[page_num]
                            # Use more aggressive text extraction
                            text += page.get_text("text", sort=True)
                        pdf_document.close()
                        
                        os.unlink(temp_file.name)
                        
                        if text.strip():
                            processing_log.append(f"Successfully extracted {len(text)} characters using PyMuPDF")
                        else:
                            error_messages.append("PyMuPDF: No text extracted")
                except Exception as e:
                    error_messages.append(f"PyMuPDF error: {str(e)}")
                    traceback.print_exc()

            # Try pdfplumber with layout analysis
            if not text:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(content)
                        temp_file.flush()
                        
                        with pdfplumber.open(temp_file.name) as pdf:
                            text = ""
                            for page in pdf.pages:
                                # Extract text with better layout preservation
                                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                                if page_text:
                                    text += page_text + "\n\n"
                                
                        os.unlink(temp_file.name)
                                
                        if text.strip():
                            processing_log.append(f"Successfully extracted {len(text)} characters using pdfplumber")
                        else:
                            error_messages.append("pdfplumber: No text extracted")
                except Exception as e:
                    error_messages.append(f"pdfplumber error: {str(e)}")
                    traceback.print_exc()

            # Try PyPDF2 as fallback
            if not text:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                        temp_file.write(content)
                        temp_file.flush()
                        
                        reader = PdfReader(temp_file.name)
                        text = ""
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n\n"
                            
                        os.unlink(temp_file.name)
                            
                        if text.strip():
                            processing_log.append(f"Successfully extracted {len(text)} characters using PyPDF2")
                        else:
                            error_messages.append("PyPDF2: No text extracted")
                except Exception as e:
                    error_messages.append(f"PyPDF2 error: {str(e)}")
                    traceback.print_exc()

            # Try OCR as last resort
            if not text:
                try:
                    processing_log.append("Attempting OCR extraction...")
                    text = extract_text_from_pdf_with_ocr(content)
                    if text.strip():
                        processing_log.append(f"Successfully extracted {len(text)} characters using OCR")
                    else:
                        error_messages.append("OCR: No text extracted")
                except Exception as e:
                    error_messages.append(f"OCR error: {str(e)}")
                    traceback.print_exc()

        text = text.strip() if text else ""
        if not text:
            # Log all processing attempts before raising the error
            print("Text extraction failed. Processing log:")
            for log in processing_log:
                print(f"- {log}")
            print("Errors encountered:")
            for error in error_messages:
                print(f"- {error}")
                
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from document. Errors encountered:\n{chr(10).join(error_messages)}"
            )

        print("Text extraction successful:")
        for log in processing_log:
            print(f"- {log}")
        
        # Initialize AI analyzer and process the text
        ai_analyzer = AIAnalyzer()
        analysis = await ai_analyzer.analyze_resume(text, job_title)
        
        return {
            "status": "success",
            "analysis": analysis,
            "extraction_method": processing_log[-1] if processing_log else "Unknown"
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the resume: {str(e)}"
        )

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
        analysis_result = analyzer.analyze(
            text=text,
            job_category=job_category,
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
