from pydantic import BaseModel

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class TranslationResponse(BaseModel):
    source_language: str
    confidence: float
    translated_text: str