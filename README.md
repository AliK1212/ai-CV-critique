# AI CV Critique

An advanced AI-powered CV analysis service that provides comprehensive feedback, ATS compatibility scoring, and industry-specific recommendations for any job type.

## Features

### 1. AI-Powered Analysis
- OPENAI integration for deep CV analysis
- Sophisticated content evaluation
- Writing style assessment
- Professional recommendations
- Industry detection and alignment scoring

### 2. Advanced Keyword Analysis
- TF-IDF based keyword extraction
- Phrase and word importance scoring
- Industry-specific keyword matching
- Skill categorization:
  - Technical skills
  - Soft skills
  - Business skills
  - Certifications
- Keyword frequency analysis
- Missing skills identification

### 3. Industry-Specific Analysis
Supports multiple industries including:
- Technology
- Finance
- Healthcare
- Marketing
- Education
- Manufacturing
- Consulting
- Retail
- Legal
- Sales
- HR
(Automatically detects industry from job title and CV content)

### 4. ATS Compatibility Analysis
- Ultra-realistic ATS scoring (0-100)
- Detailed component scoring:
  - Keyword matching (30%)
  - Format compatibility (15%)
  - Content quality (20%)
  - Readability (15%)
  - Section organization (20%)
- Format validation for ATS-friendly elements
- Comprehensive feedback system

### 5. Performance Features
- Redis caching (1-hour expiration)
- Rate limiting (5 requests/hour)
- Job-specific cache keys
- Efficient PDF text extraction with OCR backup

## Technical Stack

### Backend
- FastAPI (0.104.1)
- Python 3.9
- Document Processing (Multi-layer fallback strategy):
  - Primary: PyMuPDF (1.23.7) - Best for text-based PDFs
  - Secondary: pdfplumber (0.10.3) - Enhanced layout preservation
  - Fallback: PyPDF2 (3.0.1) - Alternative extraction method
  - OCR Support: 
    - pdf2image (1.17.0) - PDF to image conversion
    - pytesseract (0.3.10) - Text extraction from images
  - DOCX Support:
    - python-docx (0.8.11)
- NLP & ML:
  - SpaCy (3.7.2)
  - scikit-learn (1.3.2)
  - numpy (1.26.2)
- Performance:
  - Redis (5.0.1)
  - SlowAPI (0.1.8)
  - uvicorn (0.24.0)

### Frontend
- React
- TypeScript
- TailwindCSS
- Framer Motion

## Setup

1. **System Dependencies**
   ```bash
   # Ubuntu/Debian
   apt-get install -y tesseract-ocr tesseract-ocr-eng poppler-utils ghostscript
   ```

2. **Python Environment Setup**
   ```bash
   # Install Python dependencies
   pip install -r requirements.txt
   
   # Download SpaCy model
   python -m spacy download en_core_web_sm
   ```

3. **Environment Variables**
   Create a `.env` file with:
   ```
   DEEPSEEK_API_KEY=your_api_key_here
   REDIS_HOST=redis
   REDIS_PORT=6379
   ```

4. **Docker Setup**
   ```bash
   # Build and run with Docker Compose
   docker-compose up --build
   ```

## API Endpoints

### 1. CV Analysis
```http
POST /analyze-cv/
Content-Type: multipart/form-data

Parameters:
- file: PDF/DOCX file (required)
- job_title: string (optional) - for targeted analysis
```

### 2. Supported Jobs
```http
GET /supported-jobs/
Returns: List of supported job categories
```

## Response Format

```json
{
  "status": "success",
  "basic_analysis": {
    "sections_found": {},
    "contact_info": {},
    "education": [],
    "experience": [],
    "skills": {}
  },
  "ai_analysis": {
    "overall_assessment": "",
    "writing_style": "",
    "impact_score": 0,
    "ats_analysis": {
      "ats_score": 0,
      "detailed_scores": {},
      "feedback": {
        "critical": [],
        "important": [],
        "suggestions": []
      },
      "improvement_priority": []
    },
    "keyword_analysis": {
      "top_skills": [],
      "missing_skills": [],
      "skill_categories": {},
      "keyword_frequency": []
    },
    "industry_analysis": {
      "detected_industry": "",
      "industry_alignment_score": 0
    }
  }
}
```

## Architecture

```
cv-critique-api/
├── main.py              # FastAPI application & PDF processing
├── ai_analyzer.py       # AI integration & analysis
├── ats_analyzer.py      # ATS compatibility analysis
├── cv_parser.py         # CV parsing & structure analysis
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # Documentation
```

## Rate Limiting

- 5 requests per hour per IP address
- Cached results valid for 1 hour
- Separate cache keys for different job titles

## Error Handling

- Invalid file format: 400 Bad Request
- Rate limit exceeded: 429 Too Many Requests
- Server errors: 500 Internal Server Error

## Key Features for Demo

1. **Universal Job Analysis**
   - Support for PDF and DOCX formats
   - OCR backup for scanned documents
   - Automatic industry detection
   - Industry-specific recommendations

2. **Enhanced Keyword Matching**
   - TF-IDF based importance scoring
   - Phrase extraction
   - Industry-specific keyword matching
   - Skill categorization
   - Missing skills identification

3. **Visual Feedback**
   - Interactive score breakdowns
   - Color-coded recommendations
   - Progress tracking
   - Priority-based improvements

4. **Real-time Analysis**
   - Instant feedback
   - Redis caching for performance
   - Comprehensive reports
