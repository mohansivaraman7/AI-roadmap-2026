"""
OCR-BASED PDF -> FAISS PIPELINE
================================
Same end goal as your original script (PDF -> chunks -> embeddings -> FAISS),
but instead of extracting text directly (PyPDFLoader), this:

  1. Renders each PDF page as an IMAGE (using PyMuPDF)
  2. Runs OCR on that image (using pytesseract) to get text
  3. Wraps the OCR'd text back into LangChain Document objects
  4. Continues with the normal splitting -> embedding -> FAISS flow

Use this version when your PDF is a SCAN (photo/scanned image) with no
selectable text -- PyPDFLoader would return empty/garbled text in that case.

Install requirements:
    pip install pymupdf pytesseract pillow
    (also need the tesseract binary installed on the system, e.g.
     sudo apt-get install tesseract-ocr)
"""

from pathlib import Path

import pymupdf  # PyMuPDF (import name changed from `fitz` -> `pymupdf`)
import pytesseract
from PIL import Image
import io

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# CONFIGURATION
# ============================================================

PDF_FILE = "company_policy.pdf"
DB_PATH = "faiss_index_ocr"
DPI = 300  # higher DPI = sharper image = better OCR accuracy, but slower


# ============================================================
# 1. CONVERT PDF PAGES TO IMAGES, THEN OCR EACH IMAGE
# ============================================================

print("Opening PDF...")
pdf_doc = pymupdf.open(PDF_FILE)
print("PDF pages:", len(pdf_doc))

documents = []

print("Converting pages to images and running OCR...")
for page_number, page in enumerate(pdf_doc, start=1):

    # Render the page to a pixmap (image) at the given DPI.
    # zoom factor: 72 is the default PDF DPI, so DPI/72 gives the right scale
    zoom = DPI / 72
    matrix = pymupdf.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix)

    # Convert PyMuPDF pixmap -> PIL Image (what pytesseract expects)
    image_bytes = pixmap.tobytes("png")
    image = Image.open(io.BytesIO(image_bytes))

    # OPTIONAL: save the rendered page image to disk so you can inspect
    # exactly what OCR is "looking at" -- useful for debugging bad OCR output
    image_out_path = f"page_{page_number}.png"
    image.save(image_out_path)

    # Run OCR on the image to extract text
    ocr_text = pytesseract.image_to_string(image)

    print(f"  Page {page_number}: {len(ocr_text)} characters extracted via OCR")

    # Wrap into a LangChain Document, same shape PyPDFLoader would produce,
    # so the rest of the pipeline (splitter, embeddings, FAISS) needs no changes
    documents.append(
        Document(
            page_content=ocr_text,
            metadata={"source": PDF_FILE, "page": page_number}
        )
    )

pdf_doc.close()


# ============================================================
# 2. SPLIT OCR'D TEXT INTO CHUNKS
# ============================================================

print("\nSplitting OCR'd text into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# ============================================================
# 3. CREATE EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 4. CREATE FAISS DATABASE
# ============================================================

print("Creating FAISS database...")

vector_db = FAISS.from_documents(
    chunks,
    embeddings
)


# ============================================================
# 5. SAVE FAISS DATABASE
# ============================================================

vector_db.save_local(DB_PATH)

print("\n======================================")
print("FAISS DATABASE CREATED (FROM OCR)")
print("======================================")

print("Vectors:", vector_db.index.ntotal)
print("Vector dimensions:", vector_db.index.d)

print("\nDatabase location:")
print(Path(DB_PATH).absolute())

print("\nRendered page images saved as page_1.png, page_2.png, ... in the current folder.")
print("Inspect these if OCR text looks wrong -- garbled OCR is often a low-DPI or skewed-scan issue.")