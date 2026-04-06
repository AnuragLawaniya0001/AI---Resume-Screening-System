import os
import re
import pdfplumber
import pytesseract
import cv2
from PIL import Image
from docx import Document



# 1. CLEAN TEXT FUNCTION

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)  # remove extra newlines
    text = re.sub(r'[ \t]+', ' ', text)  # remove extra spaces
    text = text.strip()
    return text



# 2. EXTRACT FROM PDF

def extract_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        # If empty → scanned PDF → use OCR
        if len(text.strip()) < 50:
            print("⚠️ Scanned PDF detected → using OCR...")
            text = extract_pdf_with_ocr(file_path)

    except Exception as e:
        print("PDF Error:", e)

    return clean_text(text)



# 3. OCR FOR SCANNED PDF

def extract_pdf_with_ocr(file_path):
    text = ""
    images = []

    # Convert PDF → images using OpenCV workaround
    import fitz  # PyMuPDF
    doc = fitz.open(file_path)

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap()
        img_path = f"temp_page_{i}.png"
        pix.save(img_path)
        images.append(img_path)

    for img_path in images:
        text += extract_from_image(img_path) + "\n"
        os.remove(img_path)

    return text



# 4. EXTRACT FROM IMAGE

def extract_from_image(file_path):
    try:
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Improve OCR accuracy
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        text = pytesseract.image_to_string(gray)

    except Exception as e:
        print("Image Error:", e)
        text = ""

    return clean_text(text)



# 5. EXTRACT FROM DOCX
def extract_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print("DOCX Error:", e)

    return clean_text(text)


# -------------------------------
# 6. MAIN ROUTER FUNCTION
# -------------------------------
def extract_text(file_path):

    if not os.path.exists(file_path):
        return " File not found"

    ext = file_path.lower()

    if ext.endswith(".pdf"):
        return extract_from_pdf(file_path)

    elif ext.endswith(".docx"):
        return extract_from_docx(file_path)

    elif ext.endswith((".png", ".jpg", ".jpeg")):
        return extract_from_image(file_path)

    else:
        return "Unsupported file format"


# -------------------------------
# 7. RUN EXAMPLE
# -------------------------------
if __name__ == "__main__":

    file_path = fr"c:\Users\Anurag Lawaniya\OneDrive\Documents\Anurag_Lawaniya_Resume.pdf" 

    extracted_text = extract_text(file_path)

    print("\n EXTRACTED TEXT:\n")
    print(extracted_text)