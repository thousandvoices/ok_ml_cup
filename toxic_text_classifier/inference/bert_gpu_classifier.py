import numpy as np
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch


class BertGpuClassifier:
    def __init__(self, model_path):
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            do_lower_case=True
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            config=AutoConfig.from_pretrained(self.model_path)
        ).to('cuda')

    def predict(self, texts):
        self.model.eval()
        results = []
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer.encode(
                    text, max_length=512, truncation=True, return_tensors='pt'
                ).to('cuda')
                batch_result = self.model.forward(tokens, attention_mask=tokens > 0)[0]
                results.extend(torch.sigmoid(batch_result).detach().cpu().numpy())

        return np.float32(results)
