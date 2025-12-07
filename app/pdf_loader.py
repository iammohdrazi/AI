# app/pdf_loader.py
import pdfplumber
import os

def load_document(file_path):
    """
    Load a PDF or TXT file and return its text content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    text = ""

    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Some pages may be empty
                    text += page_text + "\n"
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        raise ValueError("Unsupported file type. Only PDF and TXT are allowed.")

    return text.strip()
