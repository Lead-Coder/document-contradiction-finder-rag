import re
import json
import hashlib
from collections import Counter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.core.config import MAX_STATEMENTS
from backend.services.context_verification import non_policy, rule_topic, features

INDEX_SCHEMA = "financial-claims-v2"


def document_metadata(pages):
    text="\n".join(t for _,t in pages)
    version=re.search(r"\bversion\s*[: ]\s*(\d+(?:\.\d+)*)",text,re.I)
    effective=re.search(r"\beffective\s*(?:date\s*[: ]\s*)?(\d{1,2}\s+\w+\s+\d{4}|\d{4}-\d{2}-\d{2})",text,re.I)
    organization=re.search(r"\b([A-Z][A-Za-z0-9 &-]{1,45}) is a fictional (?:organization|organisation|institution|company)",text)
    if not organization:
        organization=re.search(r"^\s*([A-Z][A-Za-z0-9 &-]{1,45})\s*[|]",text,re.M)
    currency="INR" if re.search(r"Indian rupees|\u20b9|\bINR\b",text) else ""
    geography="India" if re.search(r"\bin India\b",text,re.I) else ""
    return dict(version=version.group(1) if version else "",effective_date=effective.group(1) if effective else "",
        organization=organization.group(1).strip() if organization else "",currency=currency,geography=geography)


def statements(pages, document_id, filename):
    splitter=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=0,separators=["; ",", "," ",""])
    meta=document_metadata(pages)
    line_pages=Counter()
    for _,text in pages:
        line_pages.update(set(re.sub(r"\s+"," ",line).strip() for line in text.splitlines() if line.strip()))
    result=[];seen=set();section="";title="";context_only=False
    for page,text in pages:
        for block in re.split(r"\n\s*\n",text):
            parent=re.sub(r"\s+"," ",block).strip()
            lines=[]
            for line in block.splitlines():
                line=re.sub(r"\s+"," ",line).strip()
                if line.lower() in ("scope and interpretation","document control"):
                    context_only=True
                    continue
                if not line or non_policy(line):continue
                if line_pages[line]>1 and len(pages)>1 and len(line)<120 and not re.search(r"\b(?:is|are|must|may|can|shall|receive|have|has)\b|[.!?]$",line,re.I):continue
                heading=re.match(r"^(?:Section\s+)?(\d+(?:\.\d+)*)[.)]?\s+([^:]+?)(?::\s*(.*))?$",line,re.I)
                if heading and not re.search(r"[.!?]$",heading.group(2)):
                    context_only=False
                    section=heading.group(1);title=heading.group(2).strip(" -\u2014")
                    if heading.group(3):lines.append(heading.group(3).strip())
                    continue
                if context_only:continue
                # Skip short decorative headings; never discard a sentence merely mentioning policy/effective/version.
                if line.istitle() and len(line.split())<9 and not re.search(r"[.!?:]$|\b(?:is|are|must|may|can|shall|has|have)\b",line,re.I):continue
                lines.append(line)
            normalized=" ".join(lines)
            for claim in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])",normalized):
                claim=claim.strip()
                if len(claim)<12 or len(claim.split())<3 or non_policy(claim):continue
                for piece in splitter.split_documents([Document(page_content=claim)]):
                    value=piece.page_content.strip()
                    key=value.casefold()
                    if key in seen:continue
                    seen.add(key)
                    sid=hashlib.sha256(f"{INDEX_SCHEMA}:{document_id}:{page}:{value}".encode()).hexdigest()
                    row=dict(id=sid,text=value,original_text=value,document_id=document_id,filename=filename,page=page,
                        parent=parent[:6000],fragmented=len(value)<len(claim),section=section,section_title=title,
                        topic=rule_topic(value,title),index_schema=INDEX_SCHEMA,**meta)
                    f=features(row)
                    row.update(account_category=f["account_category"],features_json=json.dumps(f,ensure_ascii=True))
                    result.append(row)
                    if len(result)>MAX_STATEMENTS:raise ValueError(f"Document exceeds {MAX_STATEMENTS} claims; split the file.")
    if not result:raise ValueError("No meaningful policy claims remain after filtering headers, disclaimers and empty fragments.")
    return result
