import json
import random
import torch

from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer

random.seed(123)


def mapping(tokenized_data, head_list, tail_list):
    heads = []
    tails = []
    for index in range(len(tokenized_data['offset_mapping'])):
        for n, i in enumerate(tokenized_data['offset_mapping'][index]):
            s, e = i
            if s <= head_list[index] <= e:
                head_index = n
            if s <= tail_list[index] <= e:
                tail_index = n

            # print(tokenizer.decode(tokenized_data['input_ids'][index][head_index:tail_index]))
            # print(answer_list[index])
            # print(' ')
        heads.append(head_index)
        tails.append(tail_index)
    return heads, tails


class Dataset:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("./local_luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")

    def generate(self):
        text_list = []
        head_list = []
        tail_list = []
        question_list = []
        for name in self.config.names:
            with open('data/上行-' + name + '_labeling_result.json') as f:
                data = json.load(f)
                for i in data:
                    for j in i['data']:
                        if j['field_key'] in self.config.classes:
                            head = j['extract_info'][0]['start_offset']
                            tail = j['extract_info'][0]['end_offset']
                            if tail <= 450:
                                question_list.append(random.choice(self.config.complete_question[j['field_key']]))
                                text_list.append(i['text'][:450].lower())
                                head_list.append(head)
                                tail_list.append(tail)
        return question_list, text_list, head_list, tail_list

    def tokenize(self, question_list, text_list):
        tokenized_data = self.tokenizer(question_list,
                                        text_list,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        padding=True,
                                        truncation=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True,
                                        return_offsets_mapping=True)

        return tokenized_data

    def dataloader(self):
        question_list, text_list, head_list, tail_list = self.generate()
        tokenized_data = self.tokenize(question_list, text_list)
        heads, tails = mapping(tokenized_data, head_list, tail_list)
        # print(heads)
        # print(tokenized_data['input_ids'].shape)
        # print(len(heads))
        full_data = TensorDataset(tokenized_data['input_ids'],
                                  tokenized_data['token_type_ids'],
                                  tokenized_data['attention_mask'],
                                  torch.tensor(heads),
                                  torch.tensor(tails))

        train_size = int(0.8 * len(full_data))
        test_size = len(full_data) - train_size
        train_data, test_data = torch.utils.data.random_split(full_data, [train_size, test_size])

        train_dataloader = DataLoader(train_data, batch_size=2, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False)

        return train_dataloader, test_dataloader
