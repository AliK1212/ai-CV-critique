import os
import json
from typing import Dict, List, Set
import httpx
from ats_analyzer import ATSAnalyzer
import spacy
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import openai
from openai import AsyncOpenAI
import traceback

class AIAnalyzer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = AsyncOpenAI(api_key=api_key)
        
        self.SYSTEM_MESSAGE = """You are an expert CV analyzer. You must respond with a valid JSON object using this exact structure:
{
    "summary": "Brief overview of the CV",
    "strengths": ["List of key strengths"],
    "weaknesses": ["Areas for improvement"],
    "recommendations": ["Specific recommendations"],
    "score": "Overall score out of 100"
}
Do not include any other text or explanations outside of this JSON structure."""

        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.ats_analyzer = ATSAnalyzer()
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize TF-IDF vectorizer for keyword extraction
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
        
        # Common skills by category
        self.skill_categories = {
            'technical': set([
                'programming', 'software', 'development', 'engineering', 'analysis',
                'data', 'systems', 'design', 'testing', 'database', 'cloud',
                'architecture', 'security', 'network', 'infrastructure'
            ]),
            'soft_skills': set([
                'communication', 'leadership', 'management', 'teamwork', 'problem solving',
                'organization', 'time management', 'creativity', 'analytical', 'interpersonal',
                'collaboration', 'presentation', 'negotiation'
            ]),
            'business': set([
                'strategy', 'operations', 'marketing', 'sales', 'finance',
                'project management', 'business development', 'client relations',
                'consulting', 'planning', 'analysis', 'reporting'
            ]),
            'certifications': set([
                'certification', 'license', 'degree', 'diploma', 'certificate',
                'qualification', 'accreditation'
            ])
        }
        
        # Load industry-specific requirements
        self.industry_requirements = {
            "technology": "Software development, programming languages, technical skills",
            "finance": "Financial analysis, accounting, investment management",
            "marketing": "Digital marketing, social media, brand management",
            "healthcare": "Medical knowledge, patient care, healthcare regulations",
            "education": "Teaching experience, curriculum development, student assessment",
            "sales": "Sales techniques, customer relationship management, negotiation",
            "engineering": "Engineering principles, technical design, problem-solving",
            "human_resources": "HR policies, recruitment, employee relations",
            "data_science": "Data analysis, machine learning, statistics",
            "project_management": "Project planning, team leadership, risk management"
        }

    async def analyze_resume(self, text: str, job_title: str = None, 
                           include_industry_insights: bool = False,
                           include_competitive_analysis: bool = False,
                           detailed_feedback: bool = False) -> Dict:
        """Analyze resume text using GPT-4 model and provide comprehensive feedback."""
        try:
            if not text:
                raise ValueError("No text provided for analysis")

            print(f"Analyzing resume for job title: {job_title}")
            
            messages = [
                {
                    "role": "system",
                    "content": self.SYSTEM_MESSAGE
                },
                {
                    "role": "user",
                    "content": f"""Analyze this resume for a {job_title} position and respond with a JSON object only.

Resume text:
{text}

Analysis requirements:
- Include industry insights: {include_industry_insights}
- Include competitive analysis: {include_competitive_analysis}
- Detailed feedback: {detailed_feedback}

Remember to respond with ONLY a valid JSON object."""
                }
            ]

            try:
                response = await self.client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                print("OpenAI API Raw Response:", content)
                
                try:
                    if not content.startswith('{'):
                        raise ValueError("Response is not a JSON object")
                    analysis = json.loads(content)
                    if not isinstance(analysis, dict):
                        raise ValueError("Response is not a JSON object")
                except json.JSONDecodeError as e:
                    print(f"JSON Parse Error: {e}")
                    print(f"Raw content: {content}")
                    raise ValueError("OpenAI response was not in valid JSON format")

            except Exception as e:
                print(f"OpenAI API Error: {str(e)}")
                raise ValueError(f"Failed to get analysis from OpenAI: {str(e)}")

            # Always include ATS analysis
            ats_analyzer = ATSAnalyzer()
            ats_analysis = ats_analyzer.analyze_ats_compatibility(text)
            if 'ats_analysis' not in analysis:
                analysis['ats_analysis'] = {}
            analysis['ats_analysis'].update(ats_analysis)

            # Add industry insights if requested
            if include_industry_insights:
                industry_insights = await self._get_industry_insights(job_title)
                analysis['industry_insights'] = industry_insights

            # Add competitive analysis if requested
            if include_competitive_analysis:
                competitive_analysis = await self._get_competitive_analysis(text, job_title)
                analysis['competitive_analysis'] = competitive_analysis

            return analysis

        except Exception as e:
            print(f"Error in analyze_resume method: {e}")
            traceback.print_exc()
            raise

    async def _get_industry_insights(self, job_category: str) -> Dict:
        messages = [
            {"role": "system", "content": """You are an expert in industry analysis. You must respond with a valid JSON object using this exact structure:
{
    "trends": ["list of trends"],
    "skills": ["list of skills"],
    "salary_range": "salary information",
    "growth_opportunities": ["list of opportunities"]
}
Do not include any other text or explanations outside of this JSON structure."""},
            {"role": "user", "content": f"Analyze industry insights for {job_category} and respond with a JSON object only."}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            if not content.startswith('{'):
                raise ValueError("Response is not a JSON object")
            return json.loads(content)
        except Exception as e:
            print(f"Error getting industry insights: {e}")
            raise ValueError(f"Failed to get industry insights: {str(e)}")

    async def _get_competitive_analysis(self, resume_text: str, job_category: str) -> Dict:
        messages = [
            {"role": "system", "content": """You are an expert in competitive analysis. You must respond with a valid JSON object using this exact structure:
{
    "key_strengths": ["list of strengths"],
    "potential_gaps": ["list of gaps"],
    "unique_points": ["list of unique selling points"],
    "improvement_areas": ["list of areas to improve"]
}
Do not include any other text or explanations outside of this JSON structure."""},
            {"role": "user", "content": f"""Analyze competitive positioning for {job_category} and respond with a JSON object only.

Resume text:
{resume_text}"""}
        ]

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            if not content.startswith('{'):
                raise ValueError("Response is not a JSON object")
            return json.loads(content)
        except Exception as e:
            print(f"Error getting competitive analysis: {e}")
            raise ValueError(f"Failed to get competitive analysis: {str(e)}")

    def _detect_industry(self, job_title: str, text: str) -> str:
        """Detect the industry based on job title and resume content."""
        if not job_title:
            return None
            
        job_title_lower = job_title.lower()
        text_lower = text.lower()
        
        # Calculate industry match scores
        industry_scores = {}
        for industry, requirements in self.industry_requirements.items():
            score = 0
            # Check keywords in job title
            score += sum(1 for keyword in requirements.split(', ') 
                        if keyword.lower() in job_title_lower)
            # Check keywords in text
            score += sum(0.5 for keyword in requirements.split(', ') 
                        if keyword.lower() in text_lower)
            industry_scores[industry] = score
            
        # Return the industry with highest score
        if industry_scores:
            return max(industry_scores.items(), key=lambda x: x[1])[0]
        return None

    def _analyze_keywords(self, text: str, job_title: str = None, industry: str = None) -> Dict:
        """Enhanced keyword analysis using TF-IDF and industry context."""
        # Prepare text
        doc = self.nlp(text.lower())
        
        # Extract phrases and words
        phrases = [chunk.text for chunk in doc.noun_chunks]
        words = [token.text for token in doc if token.is_alpha and not token.is_stop]
        
        # Get TF-IDF scores
        if phrases:
            tfidf_matrix = self.tfidf.fit_transform([' '.join(phrases)])
            feature_names = self.tfidf.get_feature_names_out()
            scores = zip(feature_names, tfidf_matrix.toarray()[0])
            important_terms = sorted(scores, key=lambda x: x[1], reverse=True)[:20]
        else:
            important_terms = []

        # Identify skills
        skills_found = set()
        for category, skill_set in self.skill_categories.items():
            skills_found.update(skill for skill in skill_set 
                              if any(skill in phrase for phrase in phrases))

        # Get industry-specific requirements
        if industry and industry in self.industry_requirements:
            required_skills = set(self.industry_requirements[industry].split(', '))
            missing_skills = required_skills - skills_found
        else:
            missing_skills = set()

        return {
            "top_skills": [term[0] for term in important_terms],
            "missing_skills": list(missing_skills),
            "skill_categories": {
                category: list(skills_found & skill_set)
                for category, skill_set in self.skill_categories.items()
            },
            "keyword_frequency": Counter(words).most_common(20)
        }

    def _calculate_industry_alignment(self, text: str, industry: str) -> float:
        """Calculate how well the resume aligns with industry requirements."""
        if not industry or industry not in self.industry_requirements:
            return None
            
        requirements = self.industry_requirements[industry]
        text_lower = text.lower()
        
        # Calculate matches for each category
        scores = {
            'skills': sum(1 for skill in requirements.split(', ') 
                         if skill.lower() in text_lower) / len(requirements.split(', ')),
            'keywords': sum(1 for keyword in requirements.split(', ') 
                          if keyword.lower() in text_lower) / len(requirements.split(', ')),
            'tools': sum(1 for tool in requirements.split(', ') 
                        if tool.lower() in text_lower) / len(requirements.split(', '))
        }
        
        # Calculate weighted average
        weights = {'skills': 0.4, 'keywords': 0.3, 'tools': 0.3}
        total_score = sum(score * weights[category] 
                         for category, score in scores.items())
        
        return round(total_score * 100, 1)

    def generate_fallback_analysis(self, text: str, job_title: str = None, ats_analysis: Dict = None) -> Dict:
        """Generate basic analysis when AI service is unavailable."""
        analysis = {
            "overall_assessment": "Analysis generated using fallback system due to AI service unavailability.",
            "writing_style": "Please try again later for AI-powered analysis.",
            "impact_score": 0,
            "skills_analysis": "Service temporarily unavailable",
            "experience_analysis": "Service temporarily unavailable",
            "education_review": "Service temporarily unavailable",
            "recommendations": [
                "Try again later for AI-powered recommendations",
                "In the meantime, review your resume for clear formatting",
                "Ensure all dates and experiences are up to date"
            ]
        }
        
        if ats_analysis:
            analysis["ats_analysis"] = ats_analysis
        
        if job_title:
            analysis.update({
                "job_fit_score": 0,
                "missing_critical_skills": ["AI analysis temporarily unavailable"],
                "job_specific_recommendations": ["Please try again later for job-specific analysis"]
            })
        
        return analysis

    def generate_ats_tips(self, resume_text: str, job_title: str = None) -> List[str]:
        """Generate ATS optimization tips based on resume content and job title."""
        tips = []
        
        # Basic ATS checks
        if len(resume_text.split()) < 300:
            tips.append("Add more content to improve ATS detection of relevant keywords")
        
        if job_title and job_title.lower() in self.industry_requirements:
            reqs = self.industry_requirements[job_title.lower()]
            missing_keywords = [k for k in reqs.split(', ') if k.lower() not in resume_text.lower()]
            if missing_keywords:
                tips.append(f"Add these keywords for better ATS matching: {', '.join(missing_keywords)}")
        
        # Common ATS tips
        tips.extend([
            "Use standard section headings (e.g., 'Experience' instead of 'Career History')",
            "Avoid using tables or complex formatting",
            "Include both spelled-out and acronym versions of technical terms",
            "Use common file formats (PDF, DOCX) for submissions"
        ])
        
        return tips
