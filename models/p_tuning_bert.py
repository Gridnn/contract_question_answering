import torch
import os

from transformers import BertPreTrainedModel, BertModel, BertConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '9'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_layer = 24
num_head = 16
dim_ebd = 1024

config = BertConfig.from_pretrained("./local_luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")


class PrefixEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(15, dim_ebd)
        self.trans = torch.nn.Sequential(
            torch.nn.Linear(dim_ebd, dim_ebd),
            torch.nn.Tanh(),
            torch.nn.Linear(dim_ebd, num_layer * 2 * dim_ebd)
        ).to(device)

    def forward(self, prefix):
        prefix_tokens = self.embedding(prefix)
        past_key_values = self.trans(prefix_tokens)
        return past_key_values


class BertPrefixForQuestionAnswering(BertPreTrainedModel):
    def __init__(self):
        super().__init__(config)
        self.num_labels = 2
        self.bert = BertModel.from_pretrained("./local_luhua/chinese_pretrain_mrc_roberta_wwm_ext_large",
                                              add_pooling_layer=False)
        # self.path = "./bert-checkpoint/bert-large_best_model"
        # self.bert.load_state_dict(torch.load(self.path))
        self.qa_outputs = torch.nn.Linear(1024, self.num_labels)
        self.dropout = torch.nn.Dropout(0.3)
        self.prefix_encoder = PrefixEncoder()
        self.prefix_tokens = torch.arange(15).long()
        self.pre_seq_len = 15

        for param in self.bert.parameters():
            param.requires_grad = False

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            num_layer * 2,
            num_head,
            dim_ebd // num_head
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(self, input_ids, token_type_ids, attention_mask):
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)

        logits = self.qa_outputs(outputs[0])
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        return start_logits, end_logits
