from transformers import AutoModel, PreTrainedModel
from transformers.models.bert.configuration_bert import BertConfig
import torch
from torch import nn
import pandas as pd
from sklearn import preprocessing


class Model(PreTrainedModel):
    config_class = BertConfig

    def __init__(
        self,
        config,
        pretrained_model_name="cointegrated/rubert-tiny2",
        tokenizer=None,
        last_layer_dropout=0.2,
    ):
        super().__init__(config)
        allowed_labels = pd.read_csv("./rudoduo/labels.csv")["0"].to_list()
        self.labels_encoder = preprocessing.LabelEncoder()
        self.labels_encoder.fit(allowed_labels)

        self.tokenizer = tokenizer
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.dropout = nn.Dropout(p=last_layer_dropout)
        self.tanh = nn.Tanh()
        self.linear = nn.Linear(self.bert.config.hidden_size, len(allowed_labels))
        self.linear.weight.data.uniform_(0.0, 1.0)

    def forward(self, input_ids, return_table_embedding=False):
        bert_output = self.bert(input_ids=input_ids, return_dict=False)[0]
        output = self.dropout(bert_output)
        output = self.tanh(output)
        output = self.linear(output)
        output = output.squeeze(0)

        if len(output.shape) == 2:
            output = output.unsqueeze(0)

        cls_ids = torch.nonzero(input_ids == self.tokenizer.cls_token_id)
        filtered_logits = torch.zeros(cls_ids.shape[0], output.shape[2])

        for n in range(cls_ids.shape[0]):
            i, j = cls_ids[n]
            filtered_logits[n] = output[i, j, :]

        if return_table_embedding:
            if len(bert_output.shape) == 2:
                bert_output = bert_output.unsqueeze(0)

            filtered_logits = torch.zeros(cls_ids.shape[0], bert_output.shape[2])

            for n in range(cls_ids.shape[0]):
                i, j = cls_ids[n]
                filtered_logits[n] = bert_output[i, j, :]

            return filtered_logits
        else:
            return filtered_logits

    def predict(self, input_ids):
        model_output = self.forward(input_ids).cpu()
        predicted_labels = nn.Softmax(dim=1)(model_output).argmax(dim=1).tolist()
        return list(self.labels_encoder.inverse_transform(predicted_labels))
