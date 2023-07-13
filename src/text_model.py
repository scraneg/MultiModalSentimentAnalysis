# Text model for multi-modal sentiment analysis using BERT, only get the features from the [CLS] token

import torch.nn as nn
from transformers import BertModel, BertTokenizer


class TextModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased'):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, text, atten_masks, token_type_ids):
        # text shape: (batch_size, max_len)
        # atten_masks shape: (batch_size, max_len)
        # token_type_ids shape: (batch_size, max_len)
        output = self.bert(text, atten_masks, token_type_ids).last_hidden_state[:, 0, :]
        output = self.dropout(output)
        return output
