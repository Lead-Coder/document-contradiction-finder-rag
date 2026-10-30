import re
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.config import MAX_STATEMENTS

# Claims remain exact substrings of normalized extracted text. Long claims are
# split with the maintained LangChain splitter and marked for human review.
def statements(pages, document_id, filename):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0,
        separators=["; ", ", ", " ", ""])
    result, seen = [], set()
    for page, text in pages:
        for para in re.split(r"\n\s*\n", text):
            paragraph = re.sub(r"\s+", " ", para).strip()
            for claim in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n", paragraph):
                claim = claim.strip()
                if len(claim) < 12 or len(claim.split()) < 3:
                    continue
                pieces = splitter.split_documents([Document(page_content=claim)])
                for piece in pieces:
                    value = piece.page_content.strip()
                    key = value.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    sid = hashlib.sha256(f"{document_id}:{page}:{value}".encode()).hexdigest()
                    result.append(dict(id=sid, text=value, document_id=document_id,
                        filename=filename, page=page, parent=paragraph[:6000], fragmented=len(pieces)>1))
                    if len(result) > MAX_STATEMENTS:
                        raise ValueError(f"Document exceeds {MAX_STATEMENTS} statements; split the file.")
    if not result:
        raise ValueError("No meaningful statements found; provide longer text passages.")
    return result
