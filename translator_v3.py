import torch
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Tuple

class NLLBTranslatorV3:
    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

        # Safe limits for CPU
        self.max_tokens = 200

    # -------------------------
    # 1️⃣ TEXT CHUNKING
    # -------------------------
    def _chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids

        chunks = []
        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunks.append(
                self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            )

        return chunks

    # -------------------------
    # 2️⃣ TRANSLATE ONE CHUNK
    # -------------------------
    def _translate_chunk(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> Tuple[str, float]:

        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang),
                return_dict_in_generate=True,
                output_scores=True
            )

        translated_text = self.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )

        # -------------------------
        # 3️⃣ CONFIDENCE CALCULATION
        # -------------------------
        scores = outputs.scores
        probs = []

        for step_scores in scores:
            step_probs = torch.softmax(step_scores, dim=-1)
            max_prob = torch.max(step_probs).item()
            probs.append(max_prob)

        confidence = sum(probs) / len(probs) if probs else 0.0

        return translated_text, confidence

    # -------------------------
    # 4️⃣ PUBLIC TRANSLATE API
    # -------------------------
    def translate(
        self, text: str, src_lang: str, tgt_lang: str
    ) -> dict:

        chunks = self._chunk_text(text)

        translated_chunks = []
        confidences = []

        for chunk in chunks:
            translated, conf = self._translate_chunk(
                chunk, src_lang, tgt_lang
            )
            translated_chunks.append(translated)
            confidences.append(conf)

        final_text = " ".join(translated_chunks)
        final_confidence = sum(confidences) / len(confidences)

        return {
            "translated_text": final_text,
            "confidence": round(final_confidence, 6)
        }



if __name__ == "__main__":
    translator = NLLBTranslatorV3()

    result = translator.translate(
        text="Technology is changing the world. Artificial intelligence is powerful.",
        src_lang="eng_Latn",
        tgt_lang="hin_Deva"
    )

    print(result)
