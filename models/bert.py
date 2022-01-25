import torch
import os
import numpy as np
from transformers import BertPreTrainedModel, BertModel, BertConfig

model_path = "./local_luhua/chinese_pretrain_mrc_roberta_wwm_ext_large"
config = BertConfig.from_pretrained(model_path)


class BertPrefixForQuestionAnswering(BertPreTrainedModel):
    def __init__(self):
        super().__init__(config)
        self.num_labels = 2
        self.bert = BertModel.from_pretrained(model_path,
                                              add_pooling_layer=False)
        self.qa_outputs = torch.nn.Linear(1024, self.num_labels)

        activate_layers = np.arange(250, 390, 1)
        for n, param in enumerate(self.bert.parameters()):
            if n in activate_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, input_ids, token_type_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)

        logits = self.qa_outputs(outputs[0])
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits

    def save(self, path):
        torch.save(self.bert.state_dict(), path)
