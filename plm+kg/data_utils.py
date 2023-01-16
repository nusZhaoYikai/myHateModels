#!/usr/bin/env python
# coding=utf-8

import torch
import pandas as pd
import math
import random
from transformers import BertModel, BertTokenizer
from tqdm import trange, tqdm
import warnings
# from openprompt import PromptDataLoader
# from openprompt.data_utils import InputExample
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from knowgraph import KnowledgeGraph

warnings.filterwarnings("ignore")


def split_train_test(all_data, ratio_train=0.7, ratio_test=0.2, ratio_dev=0.1):
    '''
    默认按照 7-2-1 的比例划分训练集、测试集、验证集
    '''
    # 划分训练集、测试集、验证集
    train, middle = train_test_split(all_data, train_size=ratio_train, test_size=ratio_test + ratio_dev)
    ratio = ratio_dev / (1 - ratio_train)
    test, dev = train_test_split(middle, test_size=ratio)
    return train, test, dev

class HateDataset(Dataset):
    '''
    数据集类
    '''
    def __init__(self, args, data: pd.DataFrame, tokenizer: BertTokenizer, knowgraph=None):
        '''
        :param args:
        :param data: df columns: [id, text, label]
        :param tokenizer:
        '''
        self.tokenizer = tokenizer
        self.data = data
        self.max_seq_len = args.seq_len
        self.device = args.device
        self.knowgraph = knowgraph

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        ids = data_row['id']
        text = data_row['text']
        labels = data_row['label']

        # 注入知识图谱
        if self.knowgraph is not None:
            # 原句添加句号
            if text[-1] not in ['.', '!', '?']:
                text = text + '.'

            # 查找句中对应的三元组，并追加至句尾
            subjs, results = self.knowgraph.get(text)
            for sub, result in zip(subjs, results):
                for r in result:
                    text += ' ' + sub + ' ' + r + '.'

        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_seq_len,
                                  return_token_type_ids=False, return_attention_mask=True)

        return dict(
            ids=ids,
            labels=labels,
            input_ids=torch.LongTensor(encoding["input_ids"]).to(self.device),
            masks=torch.LongTensor(encoding["attention_mask"]).to(self.device)
        )

def load_dataset(args):
    '''
    加载数据集
    :param args:
    :param path:
    :return:
    '''
    data_path = args.data_path
    data_df = pd.read_csv(data_path + "all_data.tsv", sep='\t', header=0)
    # 随机打乱
    # data_df = data_df.sample(frac=1)
    # 列映射
    all_data = pd.DataFrame(columns=['id', 'text', 'label'])
    if "stg1" in data_path:
        label_map = {'not_hate': 0, 'implicit_hate': 1, 'explicit_hate': 2}
    elif "binary" in data_path:
        label_map = {'not_hate': 0, 'hate': 1}
    # 将标签映射为数字
    all_data['id'] = data_df['id']
    all_data['text'] = data_df['post']
    all_data['label'] = [label_map[x] for x in data_df['class']]

    # 数据增强，使得数据按照标签重复平衡
    if args.data_balance is True:
        if "stg1" in data_path:
            label0 = all_data[all_data['label'] == 0]
            label1 = all_data[all_data['label'] == 1]
            label2 = all_data[all_data['label'] == 2]
            print("[Data Balance] label0: {}, label1: {}, label2: {}".format(len(label0), len(label1), len(label2)))
            size = max(len(label0), len(label1), len(label2))
            # 重新随机采样使得数据平衡
            label1 = pd.concat([label1, label1.sample(size-len(label1), replace=True)]) # 防止随机采样中有些数据没用到
            label2 = pd.concat([label2, label2.sample(size-len(label2), replace=True)])
            all_data = pd.concat([label0, label1, label2])
        elif "binary" in data_path:
            label0 = all_data[all_data['label'] == 0]
            label1 = all_data[all_data['label'] == 1]
            print("[Data Balance] label0: {}, label1: {}".format(len(label0), len(label1)))
            size = max(len(label0), len(label1))
            label1 = pd.concat([label1, label1.sample(size - len(label1), replace=True)])
            all_data = pd.concat([label0, label1])

    # 按照随机数种子划分训练集、测试集、验证集
    train_df, test_df, dev_df = split_train_test(all_data)
    train_df.to_csv(data_path+"train.tsv", sep="\t", index=False)
    test_df.to_csv(data_path+"test.tsv", sep="\t", index=False)
    dev_df.to_csv(data_path+"dev.tsv", sep="\t", index=False)

    tokenizer = BertTokenizer.from_pretrained(args.plm_path)
    # 知识图谱工具类
    if args.use_kg is True:
        knowgraph = KnowledgeGraph(args)

    train_dataset = HateDataset(args, train_df, tokenizer, knowgraph)
    test_dataset = HateDataset(args, test_df, tokenizer, knowgraph)
    dev_dataset = HateDataset(args, dev_df, tokenizer, knowgraph)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataloader, test_dataloader, dev_dataloader