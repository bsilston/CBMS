import os
import tempfile
import zipfile
import json
from textractor import Textractor
from textractor.data.constants import TextractFeatures

def process_zip(zip_path, textractor):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract files skipping macOS metadata
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                if "__MACOSX/" in file_info.filename or file_info.filename.startswith("."):
                    continue
                zip_ref.extract(file_info, tmpdir)

        # Process all files in extracted directory
        pdf_file_paths = []
        for root, dirs, files in os.walk(tmpdir):
            for fname in files:
                full_path = os.path.join(root, fname)
                lower_name = fname.lower()
                
                if lower_name.endswith(".pdf"):
                    pdf_file_paths.append(full_path)
                elif lower_name.endswith(".zip"):
                    # Recursively process nested ZIPs
                    process_zip(full_path, textractor)

        # Process found PDFs
        for pdf_path in pdf_file_paths:
            try:
                print(f"Processing: {pdf_path}")
                document = textractor.start_document_analysis(
                    features=[TextractFeatures.TABLES],
                    file_source=str(pdf_path),
                    s3_upload_path="s3://brainmindsocietybucket",
                )
                
                # Save JSON
                json_path = f"{os.path.splitext(pdf_path)[0]}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(document.response, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"Failed to process {pdf_path}: {e}")

def main():
    # Initialize Textractor
    textractor = Textractor(profile_name="default", region_name='us-east-1')
    
    # Path to your local ZIP file
    zip_file_path = os.path.join(os.getcwd(), "CBMS test zip.zip")
    
    # Process main ZIP file
    process_zip(zip_file_path, textractor)

if __name__ == "__main__":
    main()




# with open("output.html", "w", encoding="utf-8") as f:
#     f.write(document.to_html())

# with open("output.txt", "w", encoding="utf-8") as f:
#     f.write(document.text)

# json.dumps(document.response) #to get a JSON version of the document.
# document.to_html() #to get an HTML version
# document.text


