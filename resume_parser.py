import spacy
import re
from typing import Dict, List, Optional
from datetime import datetime
from dateutil import parser as date_parser

class ResumeParser:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Common section headers
        self.section_headers = {
            'education': ['education', 'academic background', 'qualifications', 'academic history'],
            'experience': ['experience', 'work history', 'employment', 'work experience', 'professional experience'],
            'skills': ['skills', 'technical skills', 'competencies', 'expertise'],
            'projects': ['projects', 'personal projects', 'professional projects'],
            'certifications': ['certifications', 'certificates', 'accreditations'],
            'languages': ['languages', 'language proficiency'],
            'summary': ['summary', 'professional summary', 'profile', 'about me'],
            'contact': ['contact', 'contact information', 'personal information']
        }
        
        # Common skill keywords
        self.tech_skills = {
            'programming': ['python', 'java', 'javascript', 'c++', 'ruby', 'php', 'swift', 'kotlin'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask'],
            'database': ['sql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'redis'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'ai_ml': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'nlp', 'computer vision']
        }

    def extract_contact_info(self, text: str) -> Dict:
        """Extract contact information from text."""
        doc = self.nlp(text.lower())
        contact_info = {
            'email': None,
            'phone': None,
            'linkedin': None,
            'github': None
        }
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Phone pattern
        phone_pattern = r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
        phones = re.findall(phone_pattern, text)
        if phones:
            contact_info['phone'] = phones[0]
        
        # LinkedIn pattern
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        linkedin = re.findall(linkedin_pattern, text.lower())
        if linkedin:
            contact_info['linkedin'] = linkedin[0]
        
        # GitHub pattern
        github_pattern = r'github\.com/[\w-]+'
        github = re.findall(github_pattern, text.lower())
        if github:
            contact_info['github'] = github[0]
        
        return contact_info

    def extract_education(self, text: str) -> List[Dict]:
        """Extract education information."""
        education_list = []
        doc = self.nlp(text)
        
        # Common degree keywords
        degree_keywords = ['bachelor', 'master', 'phd', 'doctorate', 'bs', 'ba', 'ms', 'ma']
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(keyword in sent_text for keyword in degree_keywords):
                # Try to extract date
                dates = re.findall(r'\b(19|20)\d{2}\b', sent.text)
                year = dates[-1] if dates else None
                
                education_list.append({
                    'degree': sent.text.strip(),
                    'year': year
                })
        
        return education_list

    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience information."""
        experience_list = []
        doc = self.nlp(text)
        
        # Look for date patterns and job titles
        for sent in doc.sents:
            # Look for dates
            dates = re.findall(r'((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})', sent.text)
            
            if dates:
                experience_list.append({
                    'description': sent.text.strip(),
                    'date': dates[0]
                })
        
        return experience_list

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills and categorize them."""
        text_lower = text.lower()
        skills = {category: [] for category in self.tech_skills.keys()}
        
        for category, keywords in self.tech_skills.items():
            for skill in keywords:
                if skill in text_lower:
                    skills[category].append(skill)
        
        return {k: v for k, v in skills.items() if v}  # Remove empty categories

    def identify_sections(self, text: str) -> Dict[str, Optional[str]]:
        """Identify different sections in the resume."""
        sections = {}
        current_section = None
        section_content = []
        
        for line in text.split('\n'):
            line = line.strip().lower()
            
            # Check if line is a section header
            for section, headers in self.section_headers.items():
                if any(header in line for header in headers):
                    if current_section:
                        sections[current_section] = '\n'.join(section_content)
                    current_section = section
                    section_content = []
                    break
            else:
                if current_section:
                    section_content.append(line)
        
        # Add the last section
        if current_section:
            sections[current_section] = '\n'.join(section_content)
        
        return sections

    def analyze_resume(self, text: str) -> Dict:
        """Perform comprehensive resume analysis."""
        sections = self.identify_sections(text)
        contact_info = self.extract_contact_info(text)
        education = self.extract_education(text)
        experience = self.extract_experience(text)
        skills = self.extract_skills(text)
        
        # Generate recommendations
        recommendations = []
        
        # Check contact information
        if not contact_info['email']:
            recommendations.append("Add a professional email address")
        if not contact_info['linkedin']:
            recommendations.append("Include your LinkedIn profile")
        
        # Check sections
        if 'summary' not in sections:
            recommendations.append("Add a professional summary to highlight your key qualifications")
        if 'experience' not in sections:
            recommendations.append("Include your work experience section")
        if 'education' not in sections:
            recommendations.append("Add your educational background")
        
        # Check skills
        if not skills:
            recommendations.append("Add a detailed skills section highlighting your technical capabilities")
        
        return {
            "sections_found": {k: bool(v) for k, v in sections.items()},
            "contact_info": contact_info,
            "education": education,
            "experience": experience,
            "skills": skills,
            "recommendations": recommendations
        }
