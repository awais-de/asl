import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class TextToGlossModel(nn.Module):
    def __init__(self, model_name='t5-small'):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.
        Args:
            input_ids: input token IDs
            attention_mask: attention mask for inputs
            labels: target token IDs for teacher forcing

        Returns:
            outputs: model outputs including loss (if labels given)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs
