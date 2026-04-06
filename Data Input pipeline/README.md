# Text Extraction Module (Only Extraction)

This module is responsible for extracting raw text from different document formats.

Supported formats:
- PDF (normal + scanned)
- DOCX
- Images (JPG, PNG)

---

# Features

- Extract text from PDF using pdfplumber
- Detect scanned PDFs and apply OCR
- Extract text from images using Tesseract OCR
- Extract text from DOCX files
- Clean and normalize extracted text

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Setup (Important for OCR)

Install Tesseract OCR from:
https://github.com/tesseract-ocr/tesseract

Add to system PATH:
```
C:\Program Files\Tesseract-OCR\
```

(Optional) Add path in Python:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

# Usage

```python
from data_extract import extract_text

text = extract_text("resume.pdf")
print(text)
```

---

# Output

Returns clean raw text:

```
Arjun Sharma
arjun.sharma@email.com | +91 98765 43210 | Bangalore
...
```

---

# Files

```
data_extract.py
requirements.txt
README.md
```

---

# Notes

- Use OCR only for scanned files
- PDF extraction is faster than OCR
- Clean text before using in NLP models





