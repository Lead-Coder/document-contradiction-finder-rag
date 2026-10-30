from io import BytesIO
from pathlib import Path

def extract(content: bytes, filename: str) -> list[tuple[int, str]]:
    suffix = Path(filename).suffix.lower()
    try:
        if suffix == ".txt":
            pages = [(0, content.decode("utf-8-sig"))]
        elif suffix == ".pdf":
            import pymupdf
            with pymupdf.open(stream=content, filetype="pdf") as doc:
                if doc.needs_pass:
                    raise ValueError("Encrypted PDFs are not supported.")
                pages = [(i + 1, page.get_text(sort=True)) for i, page in enumerate(doc)]
            if not any(text.strip() for _, text in pages):
                raise ValueError("No PDF text found. Scanned PDFs require OCR, which is not supported.")
        elif suffix == ".docx":
            from docx import Document
            doc = Document(BytesIO(content))
            text = [p.text for p in doc.paragraphs]
            text += [" | ".join(cell.text for cell in row.cells) for table in doc.tables for row in table.rows]
            pages = [(0, "\n\n".join(text))]
        else:
            raise ValueError("Supported types: PDF, DOCX, TXT (UTF-8).")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot read {filename}: corrupted file or unsupported encoding.") from exc
    if not any(t.strip() for _, t in pages):
        raise ValueError("Document is empty or contains no extractable text.")
    return pages
