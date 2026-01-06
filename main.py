from fastapi import FastAPI, HTTPException
from app.translator import NLLBTranslator
from app.detector import detect_language
from app.schemas import TranslationRequest, TranslationResponse

app = FastAPI(title="NLLB One-to-One Translation API", version="1.0")

translator = NLLBTranslator()

@app.get("/")
def root():
    return {"status": "NLLB Translation API is running"}

@app.post("/translate", response_model=TranslationResponse)
def translate_text(request: TranslationRequest):
    try:
        src_lang, src_conf = detect_language(request.text)
        translated, trans_conf = translator.translate(
            request.text,
            src_lang=src_lang,
            tgt_lang=request.target_language
        )

        return TranslationResponse(
            source_language=src_lang,
            confidence=round(trans_conf, 6),
            translated_text=translated
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))