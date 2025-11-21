import pypdf
import sys

def extract_text(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    # Use the path exactly as provided by the user context, handling the space if needed
    path = "/Users/manaspokley/FINAL /Optimization_20251121180931.pdf"
    try:
        reader = pypdf.PdfReader(path)
        text = ""
        for i in range(min(5, len(reader.pages))):
            text += reader.pages[i].extract_text() + "\n"
        print(text)
    except Exception as e:
        print(e)
