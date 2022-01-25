import os
import random
import numpy as np
import torch

import config

from models.p_tuning_bert import BertPrefixForQuestionAnswering
from dataset import Dataset

random.seed(123)

os.environ['CUDA_VISIBLE_DEVICES'] = '8,9'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the loss function
def loss(start_logits, end_logits, start_index, end_index):
    total_loss = 0
    for n in range(len(start_index)):
        pred_start = start_logits[n][start_index[n]]
        pred_end = end_logits[n][end_index[n]]
        loss = - (torch.log(pred_start) + torch.log(pred_end))
    total_loss += loss
    return total_loss / len(start_index)


# define the softmax
softmax = torch.nn.Softmax(dim=-1)


def main():
    # set the cuda
    con = config.Config()
    # import the dataset
    dataset = Dataset(con)
    train_dataloader, test_dataloader = dataset.dataloader()
    # import the model
    model = BertPrefixForQuestionAnswering()
    model.to(device)
    # set the hyper-parameters
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(lr=2e-5, params=parameters)
    epochs = 20
    # train the model
    best_acc = 0
    total_loss = 0
    model.train()
    for epoch in range(epochs):
        for n, data in enumerate(train_dataloader):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)
            data[3] = data[3].to(device)
            data[4] = data[4].to(device)

            # print(tokenizer.decode(data[0][0][data[3]:data[4]]))
            start_logits, end_logits = model(data[0],
                                             token_type_ids=data[1],
                                             attention_mask=data[2])

            start_logits = softmax(start_logits)
            end_logits = softmax(end_logits)
            batch_loss = loss(start_logits, end_logits, data[3], data[4])
            # loss = outputs.loss.mean()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss
            if n != 0 and n % 20 == 0:
                print("loss: ", total_loss.cpu().item(), "batch: ", n, "/", len(train_dataloader))
                total_loss = 0

        num_true = 0
        total_num = 0
        threshold = 0.003
        maxlen = 30
        print('testing......')
        for n, data in enumerate(test_dataloader):
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            data[2] = data[2].to(device)

            start_logits, end_logits = model(data[0],
                                             token_type_ids=data[1],
                                             attention_mask=data[2])

            start_logits = softmax(start_logits)
            end_logits = softmax(end_logits)

            for a1, a2, p, true_start, ture_end in zip(start_logits, end_logits, data[0], data[3], data[4]):
                best_score = -1
                total_num += 1
                a1, a2 = a1[: len(p)], a2[: len(p)]
                l_idxs = np.where(a1.cpu() > threshold)[0]
                r_idxs = np.where(a2.cpu() > threshold)[0]
                for i in l_idxs:
                    cond = (r_idxs >= i) & (r_idxs < i + maxlen)
                    for j in r_idxs[cond]:
                        score = a1[i] * a2[j]
                        if score > best_score:
                            best_score = score
                            pred_start = i
                            pred_end = j
                p = p.cpu()
                #             print(''.join(decoder(ques.cpu())))
                #             print(''.join(decoder(p[pred_start:pred_end])))
                #             print(''.join(decoder(p[true_start:ture_end])))
                #             print(" ")
                if np.array_equal(p[pred_start:pred_end], p[true_start:ture_end]):
                    num_true += 1
        acc = num_true / total_num
        print('total answers:', total_num, '  true answers:', num_true, '  accuracy:', acc)
        # if acc > best_acc:
        #     best_acc = acc
        #     path = "./bert-checkpoint/bert-large_best_model"
        #     model.save(path)
        #     print("The best model was updated!")


if __name__ == "__main__":
    main()
