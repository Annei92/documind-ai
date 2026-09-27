from pypdf import PdfReader


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        pages.append({
            "page_number": page_number,
            "text": text or ""
        })

    return pages