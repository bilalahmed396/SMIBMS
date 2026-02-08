import torch
import torch.nn as nn
import re
from transformers import RobertaModel

class HybridBiLSTMRoBERTa(nn.Module):
    def __init__(self, roberta_model_name='roberta-base', hidden_dim=256, num_layers=2, output_dim=3):
        super(HybridBiLSTMRoBERTa, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.lstm = nn.LSTM(self.roberta.config.hidden_size, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
        
        lstm_out, _ = self.lstm(sequence_output)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

def research_clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()