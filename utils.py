import fitz  # PyMuPDF
import docx2txt
import re

def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)      # Remove multiple newlines
    text = re.sub(r'\s+', ' ', text)      # Remove extra spaces
    return text.lower()

def extract_text(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        text = extract_text_from_pdf(file_path)
    elif ext == 'docx':
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return clean_text(text)

# --- New helper functions for filtering ---
def extract_location(text):
    """Simple rule-based location extractor from resume text."""
    cities = ["Hyderabad", "Bangalore", "Bengaluru", "Pune", "Delhi", "NCR", "Mumbai", "Chennai", "Kolkata"]
    for city in cities:
        if re.search(rf"\b{city}\b", text, re.IGNORECASE):
            return city
    return "Unknown"

def extract_job_role(jd_text):
    """Naive job role extractor from JD text."""
    for line in jd_text.splitlines():
        if "role" in line.lower() or "position" in line.lower() or "title" in line.lower():
            return line.strip()
    return jd_text.splitlines()[0] if jd_text.strip() else "Unknown"
