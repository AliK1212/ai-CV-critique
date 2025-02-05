from typing import Dict, List, Tuple
import re
import spacy
from collections import Counter

class ATSAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # ATS scoring weights
        self.weights = {
            'keyword_match': 0.3,
            'format_score': 0.15,
            'content_score': 0.2,
            'readability_score': 0.15,
            'section_score': 0.2
        }
        
        # Common ATS-friendly section headers
        self.standard_sections = {
            'summary': ['summary', 'professional summary', 'profile', 'objective'],
            'experience': ['experience', 'work experience', 'professional experience', 'employment history'],
            'education': ['education', 'academic background', 'qualifications', 'academic history'],
            'skills': ['skills', 'technical skills', 'core competencies', 'expertise'],
            'certifications': ['certifications', 'certificates', 'professional certifications'],
            'projects': ['projects', 'relevant projects', 'key projects']
        }
        
        # Format checking patterns
        self.format_checks = {
            'special_chars': r'[^\w\s.,()&/-]',
            'bullet_points': r'[•·⋅‣⁃]',
            'font_markers': r'[\uFFF0-\uFFFF]|\u200B',
            'multiple_spaces': r'\s{2,}',
            'long_lines': r'.{100,}'
        }

    def analyze_ats_compatibility(self, text: str, job_description: str = None) -> Dict:
        """Perform comprehensive ATS compatibility analysis."""
        # Initialize scores
        scores = {
            'keyword_match': self._analyze_keywords(text, job_description) if job_description else 0.8,
            'format_score': self._analyze_format(text),
            'content_score': self._analyze_content(text),
            'readability_score': self._analyze_readability(text),
            'section_score': self._analyze_sections(text)
        }
        
        # Calculate weighted total score
        total_score = sum(scores[key] * self.weights[key] for key in scores)
        
        # Generate detailed feedback
        feedback = self._generate_feedback(scores, text)
        
        return {
            'ats_score': round(total_score * 100, 1),
            'detailed_scores': {k: round(v * 100, 1) for k, v in scores.items()},
            'feedback': feedback,
            'improvement_priority': self._prioritize_improvements(scores)
        }

    def _analyze_keywords(self, text: str, job_description: str) -> float:
        """Analyze keyword matching with job description."""
        if not job_description:
            return 0.8  # Default score if no job description provided
            
        # Extract key terms from job description
        job_doc = self.nlp(job_description.lower())
        job_keywords = [token.text for token in job_doc if token.pos_ in ['NOUN', 'PROPN']]
        job_keyword_freq = Counter(job_keywords)
        
        # Check resume for these keywords
        resume_doc = self.nlp(text.lower())
        resume_keywords = [token.text for token in resume_doc if token.pos_ in ['NOUN', 'PROPN']]
        resume_keyword_freq = Counter(resume_keywords)
        
        # Calculate match score
        total_keywords = len(job_keyword_freq)
        matched_keywords = sum(1 for keyword in job_keyword_freq if keyword in resume_keyword_freq)
        
        return min(matched_keywords / max(total_keywords, 1), 1.0)

    def _analyze_format(self, text: str) -> float:
        """Analyze resume formatting for ATS compatibility."""
        score = 1.0
        deductions = {
            'special_chars': 0.1,
            'bullet_points': 0.05,
            'font_markers': 0.15,
            'multiple_spaces': 0.05,
            'long_lines': 0.1
        }
        
        for check, pattern in self.format_checks.items():
            if re.search(pattern, text):
                score -= deductions[check]
        
        return max(0.0, score)

    def _analyze_content(self, text: str) -> float:
        """Analyze resume content quality."""
        doc = self.nlp(text)
        
        # Content metrics
        metrics = {
            'length': len(text.split()),
            'sentences': len(list(doc.sents)),
            'unique_words': len(set(token.text.lower() for token in doc if token.is_alpha))
        }
        
        # Score based on content metrics
        score = 1.0
        
        # Length check (ideal: 300-700 words)
        if metrics['length'] < 300:
            score -= 0.2
        elif metrics['length'] > 1000:
            score -= 0.1
            
        # Sentence variety check
        avg_sent_length = metrics['length'] / max(metrics['sentences'], 1)
        if avg_sent_length > 25:  # Too long sentences
            score -= 0.1
            
        # Vocabulary richness
        vocab_ratio = metrics['unique_words'] / max(metrics['length'], 1)
        if vocab_ratio < 0.3:  # Low vocabulary variety
            score -= 0.1
            
        return max(0.0, score)

    def _analyze_readability(self, text: str) -> float:
        """Analyze resume readability."""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if not sentences:
            return 0.0
            
        # Calculate average sentence length
        avg_sent_length = len(text.split()) / len(sentences)
        
        # Calculate readability score
        score = 1.0
        
        # Penalize for very short or very long sentences
        if avg_sent_length < 5:
            score -= 0.2
        elif avg_sent_length > 25:
            score -= 0.3
            
        # Check for passive voice
        passive_count = len([sent for sent in sentences if self._is_passive(sent)])
        passive_ratio = passive_count / len(sentences)
        if passive_ratio > 0.3:  # More than 30% passive voice
            score -= 0.2
            
        return max(0.0, score)

    def _analyze_sections(self, text: str) -> float:
        """Analyze presence and organization of standard sections."""
        text_lower = text.lower()
        score = 1.0
        
        # Check for essential sections
        essential_sections = {'experience', 'education', 'skills'}
        found_sections = set()
        
        for section_type, headers in self.standard_sections.items():
            if any(header in text_lower for header in headers):
                found_sections.add(section_type)
        
        # Penalize for missing essential sections
        missing_essential = essential_sections - found_sections
        score -= len(missing_essential) * 0.2
        
        # Check section order (preferred order: summary -> experience -> education -> skills)
        preferred_order = ['summary', 'experience', 'education', 'skills']
        current_order = []
        
        for section in preferred_order:
            if section in found_sections:
                current_order.append(section)
                
        if current_order != [s for s in preferred_order if s in current_order]:
            score -= 0.1
            
        return max(0.0, score)

    def _is_passive(self, sent) -> bool:
        """Check if a sentence is in passive voice."""
        return any(token.dep_ == 'auxpass' for token in sent)

    def _generate_feedback(self, scores: Dict[str, float], text: str) -> Dict[str, List[str]]:
        """Generate detailed feedback based on scores."""
        feedback = {
            'critical': [],
            'important': [],
            'suggestions': []
        }
        
        # Keyword matching feedback
        if scores['keyword_match'] < 0.6:
            feedback['critical'].append("Low keyword match with job requirements. Add more relevant industry terms.")
        elif scores['keyword_match'] < 0.8:
            feedback['important'].append("Consider adding more job-specific keywords to improve matching.")
            
        # Format feedback
        if scores['format_score'] < 0.7:
            feedback['critical'].append("Resume format needs improvement. Remove special characters and complex formatting.")
        
        # Content feedback
        if scores['content_score'] < 0.7:
            feedback['important'].append("Content could be more substantial. Add more specific achievements and metrics.")
            
        # Readability feedback
        if scores['readability_score'] < 0.7:
            feedback['important'].append("Improve readability by using clearer, more concise sentences.")
            
        # Section feedback
        if scores['section_score'] < 0.8:
            feedback['critical'].append("Ensure all essential sections (Experience, Education, Skills) are clearly labeled.")
            
        return feedback

    def _prioritize_improvements(self, scores: Dict[str, float]) -> List[str]:
        """Prioritize areas for improvement based on scores."""
        priorities = []
        for metric, score in sorted(scores.items(), key=lambda x: x[1]):
            if score < 0.7:
                priorities.append(f"Improve {metric.replace('_', ' ').title()}")
        return priorities
