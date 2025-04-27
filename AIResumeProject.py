#Import required libraries
import re  # Regular expressions for pattern matching
import os  # Operating system interfaces for file handling
import PyPDF2  # PDF file manipulation
import nltk  # Natural Language Toolkit for text processing
from nltk.tokenize import word_tokenize  # Text tokenization
from nltk.corpus import stopwords  # Common stopwords
import spacy  # Advanced NLP processing
from sklearn.feature_extraction.text import TfidfVectorizer  # Text vectorization
from sklearn.metrics.pairwise import cosine_similarity  # Text similarity comparison

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')  # Check if sentence tokenizer is available
except LookupError:
    nltk.download('punkt')  # Download if missing
try:
    nltk.data.find('corpora/stopwords')  # Check if stopwords are available
except LookupError:
    nltk.download('stopwords')  # Download if missing

# Load SpaCy NLP model for English language processing
try:
    nlp = spacy.load('en_core_web_sm')  # Try loading pre-installed model
except OSError:
    print("Downloading SpaCy English model...")  # Inform user if model is missing
    import subprocess
    # Download the model using Python's subprocess
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')  # Load after download

class FileHandler:
    """Handle file operations and text extraction from various formats"""
    
    @staticmethod
    def extract_text_from_file(file_path):
        """Extract text from various file formats"""
        print(f"Extracting text from file: {file_path}")  # Debug print
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if file_path.lower().endswith('.pdf'):
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text()
                return text
            except Exception as e:
                raise Exception(f"Error extracting text from PDF: {e}")
                
        elif file_path.lower().endswith('.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                raise Exception(f"Error reading text file: {e}")
                
        elif file_path.lower().endswith('.docx'):
            try:
                import docx
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                raise ImportError("python-docx package is required for DOCX files. Install it with: pip install python-docx")
            except Exception as e:
                raise Exception(f"Error extracting text from DOCX: {e}")
                
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

class JobDescriptionAnalyzer:
    """Analyze job descriptions to extract requirements and responsibilities"""
    
    def __init__(self):
        self.requirement_keywords = ['require', 'qualification', 'skill', 'experience', 'knowledge', 'proficiency']
        self.responsibility_keywords = [
            'responsibility', 'duty', 'task', 'role', 'function', 'deliver', 
            'manage', 'oversee', 'coordinate', 'lead', 'support', 'assist',
            'implement', 'develop', 'design', 'create', 'build', 'test',
            'deploy', 'maintain', 'monitor', 'report', 'document', 'analyze',
            'evaluate', 'optimize', 'troubleshoot', 'debug', 'research',
            'collaborate', 'communicate', 'present', 'train', 'mentor',
            'educate', 'advise', 'consult', 'facilitate', 'coordinate',
            'organize', 'plan', 'execute', 'strategize', 'innovate',
            'improve', 'enhance', 'streamline', 'automate', 'integrate',
            ]
        
    def extract_requirements(self, job_text):
        """Extract job requirements from job description"""
        doc = nlp(job_text)
        requirements = []
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(keyword in sent_text for keyword in self.requirement_keywords):
                requirements.append(sent.text.strip())
                
        return requirements
        
    def extract_responsibilities(self, job_text):
        """Extract job responsibilities from job description"""
        doc = nlp(job_text)
        responsibilities = []
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            print(f"Processing sentence: {sent_text}")  # Debug print
            if any(keyword in sent_text for keyword in self.responsibility_keywords):
                print(f"Matched responsibility: {sent.text.strip()}")  # Debug print
                responsibilities.append(sent.text.strip())
                
        return responsibilities
        
    def extract_job_title(self, job_text):
        """Extract potential job title from job description"""
        doc = nlp(job_text)
        first_paragraph = next(doc.sents).text
        
        # Look for potential job titles in the first paragraph
        title_patterns = [
            r'(^|\s)(Senior|Junior|Lead|Principal|Staff)?\s?(Software|Data|Web|Full Stack|Backend|Frontend)?\s?(Engineer|Developer|Scientist|Analyst|Designer)($|\s)',
            r'(^|\s)(Project|Product|Program)?\s?(Manager|Lead|Director)($|\s)',
            r'(^|\s)(UX|UI|Product|Graphic)?\s?(Designer|Architect)($|\s)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, first_paragraph, re.IGNORECASE)
            if match:
                return match.group().strip()
                
        # If no match found, return the first sentence if it's short enough
        if len(first_paragraph.split()) < 10:
            return first_paragraph
            
        return None

class ResumeAnalyzer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.skills_keywords = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'typescript', 'html', 'css', 'sql'],
            'data_science': ['machine learning', 'data analysis', 'statistics', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'r'],
            'design': ['ui/ux', 'graphic design', 'adobe', 'photoshop', 'illustrator', 'figma', 'sketch'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express', 'laravel'],
            'cloud': ['aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'devops', 'ci/cd'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'oracle', 'sql server', 'redis', 'elasticsearch'],
            'tools': ['git', 'github', 'gitlab', 'jenkins', 'jira', 'confluence', 'trello', 'slack', 'teams']
        }
        self.education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college', 'education', 'school', 'institute']
        self.experience_patterns = [r'\d+\s*years?', r'\d+\s*\+\s*years?', r'from\s+\d{4}\s+to\s+\d{4}', r'\d{4}\s*-\s*\d{4}', r'\d{4}\s*to\s*(present|current)']
        self.vectorizer = TfidfVectorizer()
        self.file_handler = FileHandler()
        self.job_analyzer = JobDescriptionAnalyzer()
        
    def extract_contact_info(self, text):
        """Extract email, phone number, LinkedIn profile and personal website"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}'
        linkedin_pattern = r'linkedin\.com/in/[A-Za-z0-9_-]+'
        website_pattern = r'https?://(?:www\.)?[A-Za-z0-9-]+\.[A-Za-z]{2,}(?:/[A-Za-z0-9-._~:/?#[\]@!$&\'()*+,;=]*)?'
        
        email = re.findall(email_pattern, text)
        phone = re.findall(phone_pattern, text)
        linkedin = re.findall(linkedin_pattern, text)
        
        # For websites, filter out LinkedIn URLs
        websites = re.findall(website_pattern, text)
        websites = [site for site in websites if 'linkedin.com' not in site]
        
        return {
            'email': email[0] if email else None,
            'phone': phone[0] if phone else None,
            'linkedin': linkedin[0] if linkedin else None,
            'website': websites[0] if websites else None
        }
    
    def extract_education(self, text):
        """Extract education information"""
        doc = nlp(text)
        education_info = []
        
        # Find sentences containing education keywords
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(keyword in sent_text for keyword in self.education_keywords):
                education_info.append(sent.text.strip())
        
        # Extract degree and university info
        degrees = []
        universities = []
        
        for edu in education_info:
            doc = nlp(edu)
            for ent in doc.ents:
                if ent.label_ == 'ORG':
                    universities.append(ent.text)
            
            degree_patterns = [
                r'(Bachelor|Master|PhD|Doctorate|BSc|MSc|BA|MA|MBA|MD|JD|BBA|BCA|MCA|MTech|BTech|BE|ME|MS)(\'s)?\s+(of|in|degree)',
                r'(Bachelor|Master|PhD|Doctorate|BSc|MSc|BA|MA|MBA|MD|JD|BBA|BCA|MCA|MTech|BTech|BE|ME|MS)(\'s)?'
            ]
            
            for pattern in degree_patterns:
                matches = re.findall(pattern, edu, re.IGNORECASE)
                if matches:
                    degrees.extend([match[0] for match in matches])
        
        return {
            'full_entries': education_info,
            'degrees': list(set(degrees)),
            'universities': list(set(universities))
        }
    
    def extract_skills(self, text):
        """Extract skills from the resume"""
        text_lower = text.lower()
        words = word_tokenize(text_lower)
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        skills = {}
        for category, keywords in self.skills_keywords.items():
            skills[category] = []
            for keyword in keywords:
                if keyword in text_lower or keyword in words:
                    skills[category].append(keyword)
        
        # Extract other potential skills (phrases with 'skill' nearby)
        doc = nlp(text)
        skill_context = []
        
        for sent in doc.sents:
            if 'skill' in sent.text.lower():
                skill_context.append(sent.text)
        
        return {
            'categorized_skills': skills,
            'skill_context': skill_context
        }
    
    def extract_experience(self, text):
        """Extract work experience information"""
        doc = nlp(text)
        experience_info = []
        
        # Find potential job titles and organizations
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                context = text[max(0, ent.start_char - 100):min(len(text), ent.end_char + 100)]
                if any(re.search(pattern, context, re.IGNORECASE) for pattern in self.experience_patterns):
                    experience_info.append({
                        'organization': ent.text,
                        'context': context
                    })
        
        # Extract job titles
        job_title_patterns = [
            r'(Senior|Junior|Lead|Principal|Staff)?\s?(Software|Data|Web|Full Stack|Backend|Frontend)?\s?(Engineer|Developer|Scientist|Analyst|Designer)',
            r'(Project|Product|Program)?\s?(Manager|Lead|Director)',
            r'(UX|UI|Product|Graphic)?\s?(Designer|Architect)'
        ]
        
        job_titles = []
        for pattern in job_title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                job_titles.extend([''.join(match).strip() for match in matches])
        
        # Extract experience duration
        years_of_experience = 0
        for pattern in self.experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches and pattern.startswith(r'\d+'):
                for match in matches:
                    num = re.search(r'\d+', match)
                    if num:
                        years_of_experience = max(years_of_experience, int(num.group()))
        
        # Extract dates to calculate experience
        date_ranges = []
        date_pattern = r'(\d{4})\s*-\s*(\d{4}|present|current)'
        date_matches = re.findall(date_pattern, text, re.IGNORECASE)
        
        for match in date_matches:
            start_year = int(match[0])
            end_year = 2025 if match[1].lower() in ['present', 'current'] else int(match[1])
            date_ranges.append((start_year, end_year))
        
        # Calculate total experience from date ranges
        if date_ranges:
            total_years = sum(end - start for start, end in date_ranges)
            years_of_experience = max(years_of_experience, total_years)
        
        return {
            'organizations': experience_info,
            'job_titles': list(set(job_titles)),
            'estimated_years': years_of_experience,
            'date_ranges': date_ranges
        }
    
    def match_job_description(self, resume_text, job_description):
        """Match resume against a job description using TF-IDF and cosine similarity"""
        print(f"Job description received: {job_description[:200]}")  # Debug print
        documents = [resume_text, job_description] # Create a list of documents for TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(documents) # Convert documents to TF-IDF matrix
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] # Calculate cosine similarity
        
        # Extract key requirements from job description
        requirements = self.job_analyzer.extract_requirements(job_description) # Extract requirements
        responsibilities = self.job_analyzer.extract_responsibilities(job_description) # Extract responsibilities
        job_title = self.job_analyzer.extract_job_title(job_description) # Extract job title
        
        # Check which skills from the job are in the resume
        all_skills = [] # Initialize empty list for all skills
        for skill_list in self.skills_keywords.values(): # Iterate through each skill category
            all_skills.extend(skill_list) # Add skills to the all_skills list
            
        job_skills = [] # Initialize empty list for job skills
        resume_skills = self.extract_skills(resume_text) # Extract skills from resume
        resume_skills_flat = [] # Initialize empty list for flat resume skills
        
        for skills in resume_skills['categorized_skills'].values(): # Iterate through each skill category
            resume_skills_flat.extend(skills) # Add skills to the flat list
            
        for skill in all_skills: # Check each skill
            if skill in job_description.lower() and skill not in resume_skills_flat: # If skill is in job description but not in resume
                job_skills.append(skill) # Add to job skills list
        
        # Calculate keyword match percentage
        keywords_in_resume = 0 # Initialize keyword match count
        job_keywords = [] # Initialize empty list for job keywords
        
        for requirement in requirements: # Extract keywords from requirements
            doc = nlp(requirement) # Process each requirement
            for token in doc: # Check each token
                if token.pos_ in ['NOUN', 'PROPN'] and token.text.lower() not in self.stop_words: # If token is a noun or proper noun and not a stop word
                    job_keywords.append(token.text.lower()) # Add to job keywords list
        
        for keyword in job_keywords: # Check each keyword
            if keyword in resume_text.lower(): # If keyword is in resume
                keywords_in_resume += 1 # Increment match count
        
        keyword_match = (keywords_in_resume / len(job_keywords)) * 100 if job_keywords else 0 # Calculate keyword match percentage
        
        return {
            'similarity_score': similarity * 100,  # Convert to percentage
            'keyword_match_score': keyword_match,
            'missing_skills': job_skills, # List of missing skills
            'key_requirements': requirements[:5],  # Limit to top 5 requirements
            'key_responsibilities': responsibilities[:5],  # Limit to top 5 responsibilities
            'job_title': job_title
        }
    
    def analyze_resume(self, resume_path, job_description=None):
        """Analyze the resume and return structured information"""
        try:
            print(f"Analyzing resume: {resume_path}")
            print(f"Job description: {job_description}")
            # Extract text from the resume file
            resume_text = self.file_handler.extract_text_from_file(resume_path)
            
            # Perform analysis
            contact_info = self.extract_contact_info(resume_text)
            education = self.extract_education(resume_text)
            skills = self.extract_skills(resume_text)
            experience = self.extract_experience(resume_text)
            
            # Calculate resume statistics
            word_count = len(word_tokenize(resume_text))
            sentence_count = len(list(nlp(resume_text).sents))
            
            analysis = {
                'contact_information': contact_info,
                'education': education,
                'skills': skills,
                'experience': experience,
                'statistics': {
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'average_sentence_length': word_count / sentence_count if sentence_count > 0 else 0
                }
            }
            
            # Analyze strengths and weaknesses
            analysis['strengths'] = []
            analysis['improvement_areas'] = []
            
            # Check technical skills coverage
            total_skills = sum(len(skills_list) for skills_list in skills['categorized_skills'].values())
            if total_skills > 10:
                analysis['strengths'].append('Strong technical skill set with diverse technologies')
            elif total_skills < 5:
                analysis['improvement_areas'].append('Limited technical skills mentioned - consider adding more specific technologies')
            
            # Check education
            if education['full_entries']:
                analysis['strengths'].append('Education background clearly presented')
            else:
                analysis['improvement_areas'].append('Education section seems missing or not clearly defined')
            
            # Check experience details
            if experience['estimated_years'] > 5:
                analysis['strengths'].append(f'Strong experience with approximately {experience["estimated_years"]} years')
            elif experience['estimated_years'] < 2:
                analysis['improvement_areas'].append('Limited work experience - consider highlighting projects or internships')
                
            # Check contact information
            missing_contact = [key for key, value in contact_info.items() if value is None]
            if missing_contact:
                analysis['improvement_areas'].append(f'Missing contact information: {", ".join(missing_contact)}')
            
            # For length and formatting
            if word_count < 300:
                analysis['improvement_areas'].append('Resume appears too short - consider adding more details')
            elif word_count > 1200:
                analysis['improvement_areas'].append('Resume may be too verbose - consider condensing information')
            
            # Match against job description if provided
            if job_description:
                match_results = self.match_job_description(resume_text, job_description)
                analysis['job_match'] = match_results
                
                if match_results['similarity_score'] > 70:
                    analysis['strengths'].append(f'Strong match with job description ({match_results["similarity_score"]:.1f}%)')
                elif match_results['similarity_score'] < 50:
                    analysis['improvement_areas'].append(f'Low match with job description ({match_results["similarity_score"]:.1f}%) - consider tailoring your resume')
                    
                if match_results['missing_skills']:
                    analysis['improvement_areas'].append(f'Missing key skills: {", ".join(match_results["missing_skills"][:5])}')
            
            return analysis
            
        except Exception as e:
            raise Exception(f"Error analyzing resume: {str(e)}")

def save_results_to_file(results, output_path):
    """Save analysis results to a file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write("RESUME ANALYSIS RESULTS\n")
            file.write("-----------------------\n\n")
            
            # Contact Information
            file.write("Contact Information:\n")
            for key, value in results['contact_information'].items():
                if value:
                    file.write(f"- {key.capitalize()}: {value}\n")
            file.write("\n")
            
            # Education
            file.write("Education:\n")
            if results['education']['degrees']:
                file.write("- Degrees: " + ", ".join(results['education']['degrees']) + "\n")
            if results['education']['universities']:
                file.write("- Universities: " + ", ".join(results['education']['universities']) + "\n")
            for edu in results['education']['full_entries']:
                file.write(f"- {edu}\n")
            file.write("\n")
            
            # Skills
            file.write("Skills:\n")
            for category, skills_list in results['skills']['categorized_skills'].items():
                if skills_list:
                    file.write(f"- {category.replace('_', ' ').capitalize()}: {', '.join(skills_list)}\n")
            file.write("\n")
            
            # Experience
            file.write("Experience:\n")
            file.write(f"- Estimated years: {results['experience']['estimated_years']}\n")
            if results['experience']['job_titles']:
                file.write("- Job Titles: " + ", ".join(results['experience']['job_titles']) + "\n")
            file.write("- Organizations:\n")
            for org in results['experience']['organizations']:
                file.write(f"  * {org['organization']}\n")
            file.write("\n")
            
            # Statistics
            file.write("Resume Statistics:\n")
            file.write(f"- Word Count: {results['statistics']['word_count']}\n")
            file.write(f"- Sentence Count: {results['statistics']['sentence_count']}\n")
            file.write(f"- Average Sentence Length: {results['statistics']['average_sentence_length']:.1f} words\n\n")
            
            # Strengths
            file.write("Strengths:\n")
            for strength in results['strengths']:
                file.write(f"- {strength}\n")
            file.write("\n")
            
            # Areas for Improvement
            file.write("Areas for Improvement:\n")
            for area in results['improvement_areas']:
                file.write(f"- {area}\n")
            file.write("\n")
            
            # Job Match if available
            if 'job_match' in results:
                file.write(f"Job Match Score: {results['job_match']['similarity_score']:.1f}%\n")
                file.write(f"Keyword Match Score: {results['job_match']['keyword_match_score']:.1f}%\n")
                
                if results['job_match']['job_title']:
                    file.write(f"Job Title: {results['job_match']['job_title']}\n")
                
                if results['job_match']['missing_skills']:
                    file.write("Missing Skills for this Job:\n")
                    for skill in results['job_match']['missing_skills']:
                        file.write(f"- {skill}\n")
                
                file.write("\nKey Job Requirements:\n")
                for req in results['job_match']['key_requirements']:
                    file.write(f"- {req}\n")
                    
                file.write("\nKey Job Responsibilities:\n")
                for resp in results['job_match']['key_responsibilities']:
                    file.write(f"- {resp}\n")
                    
        print(f"Results saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving results to file: {e}")
        return False

# Example usage
if __name__ == "__main__":
    import sys

    print("RESUME ANALYZER")
    print("--------------")

    analyzer = ResumeAnalyzer()
    results = None  # Initialize results to ensure it is always defined

    try:
        # Get the resume file path
        while True:
            resume_path = input("Enter the file path of your resume (PDF, DOCX, or TXT): ").strip('"\'')
            if os.path.exists(resume_path):
                break
            print(f"Error: File not found at path: {resume_path}")
            print("Please enter a valid file path or press Ctrl+C to exit")
            
        job_desc_method = input("Enter job description from (1) text input, (2) file, or (3) skip? Enter 1, 2, or 3: ")

        job_description = None
        if job_desc_method == "1":
            print("Enter job description text (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            job_description = '\n'.join(lines)
        elif job_desc_method == "2":
            while True:
                job_desc_path = input("Enter the file path of your job description file: ").strip('"\'')
                if os.path.exists(job_desc_path):
                    try:
                        job_description = FileHandler.extract_text_from_file(job_desc_path)
                        break
                    except Exception as e:
                        print(f"Error reading job description file: {e}")
                        print("Please try again or press Ctrl+C to exit")
                else:
                    print(f"Error: File not found at path: {job_desc_path}")
                    print("Please enter a valid file path or press Ctrl+C to exit")
        elif job_desc_method == "3":
            print("Skipping job description analysis.")
        else:
            print("Invalid option. Proceeding without job description comparison.")

        # Debug prints
        print(f"Analyzing resume: {resume_path}")
        print(f"Job description: {job_description}")

        # Analyze the single resume
        try:
            results = analyzer.analyze_resume(resume_path, job_description)
        except Exception as e:
            print(f"Error during resume analysis: {e}")
            import traceback
            traceback.print_exc()
            results = None

    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    if results is None:
        print("No results to display due to an error during analysis.")
    else:
        # Save or display results
        save_option = input("Would you like to save results to a file? (y/n): ")
        if save_option.lower() == 'y':
            output_path = input("Enter file path for output (default: resume_analysis.txt): ") or "resume_analysis.txt"
            save_results_to_file(results, output_path)
        else:
            # Print results in a readable format
            print("\nRESUME ANALYSIS RESULTS")
            print("-----------------------")
            print("\nContact Information:")
            for key, value in results['contact_information'].items():
                if value:
                    print(f"- {key.capitalize()}: {value}")
            
            print("\nEducation:")
            if results['education']['degrees']:
                print("- Degrees: " + ", ".join(results['education']['degrees']))
            if results['education']['universities']:
                print("- Universities: " + ", ".join(results['education']['universities']))
            for edu in results['education']['full_entries']:
                print(f"- {edu}")
            
            print("\nSkills:")
            for category, skills_list in results['skills']['categorized_skills'].items():
                if skills_list:
                    print(f"- {category.replace('_', ' ').capitalize()}: {', '.join(skills_list)}")
            
            print("\nExperience:")
            print(f"- Estimated years: {results['experience']['estimated_years']}")
            if results['experience']['job_titles']:
                print("- Job Titles: " + ", ".join(results['experience']['job_titles']))
            print("- Organizations:")
            for org in results['experience']['organizations']:
                print(f"  * {org['organization']}")
            
            print("\nResume Statistics:")
            print(f"- Word Count: {results['statistics']['word_count']}")
            print(f"- Sentence Count: {results['statistics']['sentence_count']}")
            print(f"- Average Sentence Length: {results['statistics']['average_sentence_length']:.1f} words")
            
            print("\nStrengths:")
            for strength in results['strengths']:
                print(f"- {strength}")
            
            print("\nAreas for Improvement:")
            for area in results['improvement_areas']:
                print(f"- {area}")
            
            if 'job_match' in results:
                print(f"\nJob Match Score: {results['job_match']['similarity_score']:.1f}%")
                print(f"Keyword Match Score: {results['job_match']['keyword_match_score']:.1f}%")
                
                if results['job_match']['job_title']:
                    print(f"Job Title: {results['job_match']['job_title']}")
                
                if results['job_match']['missing_skills']:
                    print("\nMissing Skills for this Job:")
                    for skill in results['job_match']['missing_skills']:
                        print(f"- {skill}")
                    
                print("\nKey Job Requirements:")
                for req in results['job_match']['key_requirements']:
                    print(f"- {req}")
                    
                print("\nKey Job Responsibilities:")
                if results['job_match']['key_responsibilities']:
                    for resp in results['job_match']['key_responsibilities']:
                        print(f"- {resp}")
                else:
                    print("No responsibilities found in the job description.")
                    
#C:\Users\adgib\Downloads\AnilaGibsonResume2025.pdf
"""Job description
About Boston Bioprocess

Boston Bioprocess, Inc.’s mission is to help fermentation product companies develop and scale up production. We specialize in providing innovative solutions and services to support the research, development, and manufacturing processes of our clients.

About The Role

We are seeking a highly motivated software engineering intern for Summer of 2025 to work and contribute to our growing software stack
We are a startup! Prospective candidates should enjoy working in a small team with minimal bureaucracy.

What You'll Do

Assist our software team in improving our database registration and organization architecture
Work with our software and machine-learning engineers to build data dashboards extracting information from our LIMS database
Develop data parsing and data analysis packages to parse unstructured scientific data from laboratory experiments and scientific literature
Assist with other projects and tasks as necessary

Qualifications

Education:
Final-year undergraduate or master's student in Computer Science or related field
Technical skills:
Experience in full-stack development with Node.js, FastAPI, React, TailwindCSS, Plotly.js
Proficiency in databases: Postgre SQL, SQL, NoSQL
Familiarity with Alembic for database migrations
Strong API development and API testing skills
Ability to debug code efficiently
Experience in dashboard creation and data visualization
Familiarity with version control tools like Git
UI/UX & Product Ownership:
A good eye for UI/UX design and detail-oriented approach
Ability to take ownership of features and drive improvements
AI & Efficiency:
Ability to leverage AI tools to write minimal, efficient code
Understanding of AI-generated code and capability to debug/fix issues when required
Work Ethic & Growth Mindset:
Eager to learn and grow in a fast-paced startup environment
Good academic standing with a strong problem-solving mindset
Excellent problem-solving skills and attention to detail
Strong communication and teamwork abilities"""
#C:\Users\adgib\Downloads\resume_analysis.txt