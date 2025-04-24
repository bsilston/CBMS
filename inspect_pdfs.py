import os
from PyPDF2 import PdfReader

def print_first_page_lines(pdf_folder):
    for fname in os.listdir(pdf_folder):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(pdf_folder, fname)
            try:
                reader = PdfReader(path)
                first_page_text = reader.pages[0].extract_text() or ""
                lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
                
                print(f"\n📄 PDF: {fname}")
                print("---- First Page Lines ----")
                for i, line in enumerate(lines):
                    print(f"{i + 1:02d}: {line}")
            except Exception as e:
                print(f"❌ Error reading {fname}: {e}")

# 🔧 Update this path to the folder where your PDFs are stored
print_first_page_lines("/Users/briansilston/Dropbox/Neuroscience General/Columbia Center/RAG/data/CBMS test zip")