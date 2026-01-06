from langdetect import detect_langs

LANG_MAP = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn"
}


def detect_language(text: str):
    results = detect_langs(text)
    top = results[0]
    lang = str(top.lang)
    confidence = top.prob

    return LANG_MAP.get(lang, "eng_Latn"), confidence