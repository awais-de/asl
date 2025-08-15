# model.py
import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class TextToGlossModel(nn.Module):
    def __init__(self, model_name='t5-small', device=None):
        super().__init__()
        self.model_name = model_name
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def predict_glosses(self, text_list, max_length=50):
        self.model.eval()
        encoded = self.tokenizer(text_list, return_tensors='pt', padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            input_ids=encoded['input_ids'],
            attention_mask=encoded['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
