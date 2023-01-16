"""
使用bert进行文本分类
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch.optim import AdamW
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

data_dir = "./data/implicit-hate-corpus-v1/stg1"
train_file = os.path.join(data_dir, "train.tsv")
dev_file = os.path.join(data_dir, "dev.tsv")
test_file = os.path.join(data_dir, "test.tsv")


class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.file_path = file_path
        self.data = self.read_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def read_data(self):
        data = pd.read_csv(self.file_path, sep='\t', header=0)
        data = data.values.tolist()
        output = []
        for id, text, label in data:
            text = self.tokenizer.encode_plus(text, max_length=self.max_len, padding='max_length', truncation=True)
            output.append((torch.tensor(text['input_ids']), torch.tensor(text['attention_mask']), torch.tensor(label)))
        return output


def get_dataloader(batch_size=768):
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    print(f"vocab_size: {tokenizer.vocab_size}")
    train_dataset = MyDataset(train_file, tokenizer)
    dev_dataset = MyDataset(dev_file, tokenizer)
    test_dataset = MyDataset(test_file, tokenizer)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader, test_loader


class CNNModel(nn.Module):
    def __init__(self, vocab_size, static=True):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 768)
        kernel_sizes = [2, 3, 4]
        kernel_num = 100
        # kernel_num等于bert的tokenizer的vocab_size
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, kernel_num, (k, 768)) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_num, 3)
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x, att, label=None):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = F.softmax(self.fc(x), -1)

        if label is not None:
            loss = self.criteria(logit.view(-1, 3), label.view(-1))
            return loss, logit
        else:
            return None, logit


def train(args, model, train_loader, dev_loader, optimizer, device, epoch, save_path="./out_models"):
    model.train()
    for i, (input_ids, attention_mask, label) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        loss, logits = model(input_ids, attention_mask, label=label)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("epoch: {} step: {} loss: {}".format(epoch, i, loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        f1 = 0
        presion = 0
        recall = 0
        pbar = tqdm(dev_loader)
        for i, (input_ids, attention_mask, label) in enumerate(pbar):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            _, logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1)
            correct += torch.sum(pred == label).item()
            total += len(label)
            f1 += f1_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')
            presion += precision_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')
            recall += recall_score(label.cpu().numpy(), pred.cpu().numpy(), average='macro')
            pbar.set_description("acc: {} f1: {} presion: {} recall: {}".format(correct / total, f1 / (i + 1),
                                                                                presion / (i + 1), recall / (i + 1)))
        final_acc = correct / total
        final_f1 = f1 / (i + 1)
        final_presion = presion / (i + 1)
        final_recall = recall / (i + 1)
        eval_result = {
            'acc': final_acc,
            'f1': final_f1,
            'precision': final_presion,
            'recall': final_recall
        }
        print(eval_result)
        """
        在best_result.json中记录最好的实验结果
        """

        # 以json文件的形式保存每一次的实验结果
        if not os.path.exists(f'./log/best_{args.model_name}_result.json'):
            with open(f'./log/best_{args.model_name}_result.json', 'w') as f:
                json.dump(eval_result, f)
        else:
            # 保存每一次的实验结果
            with open(f'./log/best_{args.model_name}_result.json', 'r') as f:
                old_result = json.load(f)
            # 比较新旧实验结果，如果新实验结果更好，则保存新实验结果
            if eval_result["f1"] > old_result["f1"]:
                # 保存模型checkpoint到args.save_path路径下
                torch.save(model.state_dict(), save_path + f'best_{args.model_name}_model.pt')
                # 保存新实验结果
                with open(f'./log/best_{args.model_name}_result.json', 'w') as f:
                    json.dump(eval_result, f)


def predict(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        preds = []
        pbar = tqdm(test_loader)
        labels = []
        for i, (input_ids, attention_mask, label) in enumerate(pbar):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # label = label.to(device)
            _, logits = model(input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy().tolist())
            labels.extend(label.cpu().numpy().tolist())
    # 计算acc\presion\recall\f1
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    presion = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    print("final test acc: {} f1: {} presion: {} recall: {}".format(acc, f1, presion, recall))


def main():
    passer = argparse.ArgumentParser()
    passer.add_argument('--batch_size', type=int, default=32)
    passer.add_argument('--lr', type=float, default=1e-5)
    passer.add_argument('--epochs', type=int, default=20)
    passer.add_argument('--save_path', type=str, default='./out_models/')
    passer.add_argument('--model_name', choices=["baseline_bert", "cnn", "lstm"], default="cnn", type=str,
                        help="模型名")

    args = passer.parse_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists('./log'):
        os.mkdir('./log')

    # 设置随机种子
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed_all(2020)
    # 加载数据
    print("开始加载数据")
    t1 = time.time()
    train_loader, dev_loader, test_loader = get_dataloader(args.batch_size)
    print(f"加载数据完成, 耗时{time.time() - t1}s")
    # 加载模型,vocab_size等于BertTokenizer的vocab_size
    print("开始加载模型")
    vocab_size = 30522
    model = CNNModel(vocab_size)
    # 设置GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 设置优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # 训练模型
    print("开始训练模型")
    t2 = time.time()
    for epoch in range(args.epochs):
        train(args, model, train_loader, dev_loader, optimizer, device, epoch)
    print(f"训练模型完成, 耗时{time.time() - t2}s")
    # 预测
    save_path = './out_models/'
    if os.path.exists(save_path + f'best_{args.model_name}_model.pt'):
        model.load_state_dict(torch.load(save_path + f'best_{args.model_name}_model.pt'))
    # model.load_state_dict(torch.load(save_path + f'best_baseline_bert_model.pt'))
    print("开始预测")
    predict(model, test_loader, device)
    print(f"预测完成,总耗时{time.time() - t1}s")


if __name__ == '__main__':
    main()
