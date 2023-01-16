#!/usr/bin/env python
# coding=utf-8

import warnings

import torch
import torch.nn as nn
from transformers import BertModel

# from openprompt import PromptForClassification
# from openprompt.plms import load_plm
# from openprompt.prompts import ManualTemplate
# from openprompt.prompts import ManualVerbalizer


warnings.filterwarnings("ignore")


class HateClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.plm_path, return_dict=True)
        self.gru = nn.GRU(input_size=self.bert.config.hidden_size, hidden_size=512, num_layers=5,
                          dropout=0.3, batch_first=True, bidirectional=True)

        if "binary" in args.data_path:
            label_num = 2
        elif "stg1" in args.data_path:
            label_num = 3

        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, label_num)

        self.fc = nn.Linear(768, label_num)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask).pooler_output

        # # TODO 1. 尝试 LSTM 2. 尝试用 last_hidden_state 做分类
        # outputs = outputs.view(outputs.shape[0], 1, -1)
        # outputs, _ = self.gru(outputs)
        # outputs = self.fc2(self.fc1(outputs))
        # logits = torch.sigmoid(outputs).view(logits.shape[0], -1)

        outputs = self.fc(outputs)
        logits = torch.softmax(outputs, axis=-1)
        return logits
