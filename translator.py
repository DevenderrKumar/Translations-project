import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class NLLBTranslator:
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

    def translate(self, text: str, src_lang: str, tgt_lang: str):
        self.tokenizer.src_lang = src_lang

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                output_scores=True,
                return_dict_in_generate=True,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_lang)
            )

        decoded = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        # Phase 3 confidence calculation
        scores = outputs.scores
        token_probs = [torch.softmax(score, dim=-1).max().item() for score in scores]
        confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

        return decoded, confidence