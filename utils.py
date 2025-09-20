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