from typing import Literal
from pydantic import BaseModel, Field

class CompareRequest(BaseModel):
    document_ids: list[str] = Field(min_length=2, max_length=20)
    threshold: float = Field(default=.75, ge=.5, le=1)
    top_k: int = Field(default=5, ge=1, le=50)
    bidirectional: bool = True

class EvaluationRequest(BaseModel):
    dataset: Literal["multinli", "snli", "documents"] = "multinli"
    split: Literal["validation_matched", "validation_mismatched", "validation"] = "validation_matched"
    count: int = Field(default=300, ge=3, le=1000)
    threshold: float = Field(default=.75, ge=.5, le=1)

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    document_ids: list[str] = Field(default_factory=list)

class DocumentInfo(BaseModel):
    id: str
    filename: str
    filetype: str
    statements: int
    preview: str

class JobInfo(BaseModel):
    id: str
    status: str
    progress: float
    message: str
    result: dict | None = None
    error: str | None = None
