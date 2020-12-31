import numpy as np
import onnxruntime
from pathlib import Path
import gzip
from transformers import AutoTokenizer
import re


class BertCpuClassifier:
    def __init__(self, model_path):
        self.model_path = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            do_lower_case=True
        )

        onnx_model_path = self.onnx_model_path(self.model_path)
        with gzip.open(onnx_model_path, 'rb') as model_file:
            self.session = onnxruntime.InferenceSession(
                model_file.read(),
                onnxruntime.SessionOptions(),
                providers=['CPUExecutionProvider']
            )

        self.emoji_pattern = re.compile(
            '['
            u'\U0001F600-\U0001F64F'
            u'\U0001F300-\U0001F5FF'
            u'\U0001F680-\U0001F6FF'
            u'\U0001F1E0-\U0001F1FF'
            ']+', flags=re.UNICODE
        )

    @staticmethod
    def onnx_model_path(path):
        return Path(path) / 'model.onnx.gz'

    def predict(self, texts):
        results = []
        for text in texts:
            text = self.emoji_pattern.sub(r' ', text.strip())
            tokens = self.tokenizer.encode(
                text, max_length=512, truncation=True, return_tensors='np')

            ort_inputs = {
                'input_ids':  tokens,
                'attention_mask':  (tokens > 0).astype(np.int64),
                'token_type_ids':  np.zeros_like(tokens, dtype=np.int64)
            }
            batch_result = self.session.run(None, ort_inputs)[0]
            results.extend(batch_result)

        results = np.float32(results)
        return 1 / (1 + np.exp(-results))
