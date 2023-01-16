# -*- encoding:utf-8 -*-
"""
  This script provides an k-BERT exmaple for classification.
"""
import argparse
import os
import random
import sys
import time
import torch
import torch.nn as nn
import warnings
from multiprocessing import Pool

from brain import KnowledgeGraph
from parser import *
from uer.model_builder import build_model
from uer.model_saver import save_model
from uer.utils.config import load_hyperparam
from uer.utils.constants import *
from uer.utils.optimizers import BertAdam
from uer.utils.seed import set_seed
from uer.utils.tokenizer import SpacyTokenizer, BertTokenizer
from uer.utils.vocab import Vocab

warnings.filterwarnings('ignore')


# 加入 LSTM
class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.args = args

        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling

        self.output_layer = nn.Linear(args.hidden_size, args.labels_num)

        self.softmax = nn.Softmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, postag_ids, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """

        if self.args.encoder == "bert":
            emb = self.embedding(src, postag_ids, mask, pos)

        else:
            emb = self.embedding(src)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        logits = self.output_layer(output)  # [batch_size x labels_num]
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits


def add_knowledge_worker(params, postag_dict):
    p_id, sentences, columns, kg, vocab, args = params
    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 1000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush()
        line = line.strip().split('\t')

        label = int(line[columns["label"]])

        text = CLS_TOKEN + ' ' + line[columns["text"]]

        tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
        tokens = tokens[0]
        pos = pos[0]
        vm = vm[0].astype("bool")

        token_ids = [vocab.get(t) for t in tokens]

        mask = [1 if t != PAD_TOKEN else 0 for t in tokens]

        # postag 信息
        if args.use_postag:

            postag_ids = [args.postag_vocab.get(t) for t in postag_dict[str(line_id + 1)]]
            postag_ids = postag_ids + [0] * (args.seq_length - len(postag_ids))

            dataset.append((token_ids, postag_ids, label, mask, pos, vm))
        else:
            dataset.append((token_ids, label, mask, pos, vm))

    return dataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_model_path", default="./models/classifier_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--dataset_path", default="./data/implicit-hate-corpus-v1/stg1/")
    parser.add_argument("--train_path", type=str, required=False,
                        default="./data/implicit-hate-corpus-v1/stg1/train.tsv",
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=False, default="./data/implicit-hate-corpus-v1/stg1/dev.tsv",
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=False, default="./data/implicit-hate-corpus-v1/stg1/test.tsv",
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", default="bert", help="Encoder type.")

    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last", "all"], default="all",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["spacy", "bert", "char", "word", "space"], default="spacy",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate. eg: 2e-5")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=1,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=3047,
                        help="Random seed.")

    # Pretrain options
    parser.add_argument("--pretrain_data_path", default="data/pretrain_data.txt")
    parser.add_argument("--checkpoint_save_path", type=str, default="./models/checkpoint")
    # parser.add_argument("--pretrain_save_path", type=str, default="./models/bert-base-patent")
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=5)  # 限制checkpoints的数量，最多5个

    # Evaluation options.
    # parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA dataset.")

    # kg
    parser.add_argument("--kg_name", required=False, default="brain/kgs/hate_brains.spo", help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading dataset")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    # extra options
    parser.add_argument("--use_postag", action="store_true", help="Use postag.")
    parser.add_argument("--save_path", type=str, default="./out_models/", help="Path of the output model.")

    args = parser.parse_args()
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)
    args.output_model_path = f"./models/{args.encoder}_model.bin"
    set_seed(args.seed)

    # Count the number of labels.
    labels_set = dict()
    columns = {}
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                if label not in labels_set:
                    labels_set[label] = 0
                labels_set[label] += 1
                # labels_set.add(label)
            except:
                pass
    args.labels_num = len(labels_set.keys())
    print("labels_num:", args.labels_num)
    for label, count in labels_set.items():
        print("label: %d, count: %d" % (label, count))
    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    # 除了加载原词表外，将数据集和知识图谱中的词也加入词表
    from uer.utils.tokenizer import BertTokenizer
    tokenizer = SpacyTokenizer(args) if args.tokenizer == "spacy" else BertTokenizer(args)
    vocab.union(args.train_path, tokenizer, min_count=2)
    vocab.union(args.dev_path, tokenizer, min_count=2)
    vocab.union(args.test_path, tokenizer, min_count=2)
    vocab.union(args.kg_name, tokenizer, min_count=1, type='kg')
    args.vocab = vocab
    # 词性标注词表
    # parse(args)
    postag_vocab = Vocab()
    postag_vocab.load(args.dataset_path + 'postag_vocab.txt')
    args.postag_vocab = postag_vocab
    # 位置词表
    pos_vocab = Vocab()
    pos_vocab.load(args.dataset_path + 'pos_vocab.txt')
    args.pos_vocab = pos_vocab

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # 预训练模型
    # model = pre_train(args, model)

    # Build classification model.
    model = BertClassifier(args, model)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None and os.path.exists(args.pretrained_model_path):
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)

    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)

    # Datset loader.
    def batch_loader(batch_size, input_ids, postag_ids, label_ids, mask_ids, pos_ids, vms):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i * batch_size: (i + 1) * batch_size, :]
            postag_ids_batch = postag_ids[i * batch_size: (i + 1) * batch_size, :] if postag_ids is not None else None
            label_ids_batch = label_ids[i * batch_size: (i + 1) * batch_size]
            mask_ids_batch = mask_ids[i * batch_size: (i + 1) * batch_size, :]
            pos_ids_batch = pos_ids[i * batch_size: (i + 1) * batch_size, :]
            vms_batch = vms[i * batch_size: (i + 1) * batch_size]
            yield input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num // batch_size * batch_size:, :]
            postag_ids_batch = postag_ids[instances_num // batch_size * batch_size:,
                               :] if postag_ids is not None else None
            label_ids_batch = label_ids[instances_num // batch_size * batch_size:]
            mask_ids_batch = mask_ids[instances_num // batch_size * batch_size:,
                             :]
            pos_ids_batch = pos_ids[instances_num // batch_size * batch_size:,
                            :]
            vms_batch = vms[instances_num // batch_size * batch_size:]

            yield input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(args, spo_files=spo_files, predicate=True)

    def read_dataset(path, workers_num=1):

        print("Loading sentences from {}".format(path))
        sentences = []
        with open(path, mode='r', encoding="utf-8") as f:
            for line_id, line in enumerate(f):
                if line_id == 0 or line == "\n":
                    continue
                sentences.append(line)
        sentence_num = len(sentences)

        # 读取语法信息
        with open(path.replace('.tsv', '_postag.json'), 'r') as f:
            json_dict = json.load(fp=f)

        print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(
            sentence_num, workers_num))
        if workers_num > 1:
            params = []
            sentence_per_block = int(sentence_num / workers_num) + 1
            for i in range(workers_num):
                params.append(
                    (i, sentences[i * sentence_per_block: (i + 1) * sentence_per_block], columns, kg, ƒ, args))
            pool = Pool(workers_num)
            res = pool.map(add_knowledge_worker, params, json_dict)
            pool.close()
            pool.join()
            dataset = [sample for block in res for sample in block]
        else:
            params = (0, sentences, columns, kg, vocab, args)
            dataset = add_knowledge_worker(params, json_dict)

        # 将数据打乱
        random.shuffle(dataset)

        return dataset

    # Evaluation function.
    def evaluate(args, is_test, metrics='Acc'):

        metrics = metrics.lower()  # 保证小写

        if is_test:
            dataset = read_dataset(args.test_path, workers_num=args.workers_num)
        else:
            dataset = read_dataset(args.dev_path, workers_num=args.workers_num)
        if args.use_postag:
            input_ids = torch.LongTensor([sample[0] for sample in dataset])
            postag_ids = torch.LongTensor([sample[1] for sample in dataset])
            label_ids = torch.LongTensor([sample[2] for sample in dataset])
            mask_ids = torch.LongTensor([sample[3] for sample in dataset])
            pos_ids = torch.LongTensor([example[4] for example in dataset])
            vms = [example[5] for example in dataset]
        else:
            input_ids = torch.LongTensor([sample[0] for sample in dataset])
            postag_ids = None
            label_ids = torch.LongTensor([sample[1] for sample in dataset])
            mask_ids = torch.LongTensor([sample[2] for sample in dataset])
            pos_ids = torch.LongTensor([example[3] for example in dataset])
            vms = [example[4] for example in dataset]

        batch_size = args.batch_size
        instances_num = input_ids.size()[0]
        if is_test:
            print("The number of evaluation instances: ", instances_num)

        correct = 0
        # Confusion matrix.
        confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

        model.eval()

        for i, (input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch,
                vms_batch) in enumerate(
            batch_loader(batch_size, input_ids, postag_ids, label_ids, mask_ids, pos_ids, vms)):

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            postag_ids_batch = postag_ids_batch.to(device) if postag_ids_batch is not None else None
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            with torch.no_grad():
                try:
                    loss, logits = model(input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch,
                                         pos_ids_batch, vms_batch)
                except Exception as e:
                    print(e)
                    print(input_ids_batch)
                    print(input_ids_batch.size())
                    print(vms_batch)
                    print(vms_batch.size())

            logits = nn.Softmax(dim=1)(logits)
            pred = torch.argmax(logits, dim=1)
            gold = label_ids_batch
            # print(f"pred: {pred}\t gold: {gold}")
            # print(pred)
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred == gold).item()

        if is_test:
            print("Confusion matrix:")
            print(confusion)
            print("Report precision, recall, and f1:")

        all_precision = 0
        all_recall = 0
        all_f1 = 0

        for i in range(confusion.size()[0]):
            try:  # 防止除 0 错误
                precision = confusion[i, i].item() / confusion[i, :].sum().item()
            except:
                precision = 0
            try:
                recall = confusion[i, i].item() / confusion[:, i].sum().item()
            except:
                recall = 0
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except:
                f1 = 0
            # if i == 1:
            #     label_1_f1 = f1
            print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, precision, recall, f1))
            all_precision += precision
            all_recall += recall
            all_f1 += f1

        final_f1 = all_f1 / confusion.size()[0]
        final_precision = all_precision / confusion.size()[0]
        final_recall = all_recall / confusion.size()[0]
        final_acc = correct / instances_num

        print('##########')
        if is_test:
            print("Test result:")
        else:
            print("Dev result:")
        print("Acc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
        print("F1. {}".format(final_f1))
        print('precision: {}'.format(final_precision))
        print('Recall: {}'.format(final_recall))
        print('##########')

        # 保存本次实验的结果
        # if not os.path.exists('./log/result.txt'):
        #     os.mknod('./log/result.txt')

        eval_result = {
            'acc': final_acc,
            'f1': final_f1,
            'precision': final_precision,
            'recall': final_recall
        }
        """
        在result.txt中记录每一次实验结果
        在best_result.json中记录最好的实验结果
        """

        # 记录此次实验的结果到result.txt文件中
        with open('./log/result.txt', 'a') as f:
            # 写入实验时间
            # f.write("#" * 30 + "\n")
            f.write("test" + " result:\n" if is_test else "")
            f.write('time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            for key, value in eval_result.items():
                f.write('{}: {}\t'.format(key, value))
            f.write('\n')
            # f.write("#" * 30 + "\n")

        # 以json文件的形式保存每一次的实验结果
        if not os.path.exists('./log/best_result.json'):
            with open('./log/best_result.json', 'w') as f:
                json.dump(eval_result, f)
        else:
            # 保存每一次的实验结果
            with open('./log/best_result.json', 'r') as f:
                old_result = json.load(f)
            # 比较新旧实验结果，如果新实验结果更好，则保存新实验结果
            if eval_result[metrics] > old_result[metrics]:
                # 保存模型checkpoint到args.save_path路径下
                torch.save(model.state_dict(), args.save_path + 'best_model.pt')
                # 保存新实验结果
                with open('./log/best_result.json', 'w') as f:
                    json.dump(eval_result, f)

        if metrics == 'f1':
            return final_f1
        else:
            return final_acc

    # Training phase.
    print("Start training.")
    trainset = read_dataset(args.train_path, workers_num=args.workers_num)
    instances_num = len(trainset)
    batch_size = args.batch_size

    print("Trans data to tensor.")
    if args.use_postag:
        input_ids = torch.LongTensor([example[0] for example in trainset])
        postag_ids = torch.LongTensor([example[1] for example in trainset])
        label_ids = torch.LongTensor([example[2] for example in trainset])
        mask_ids = torch.LongTensor([example[3] for example in trainset])
        pos_ids = torch.LongTensor([example[4] for example in trainset])
        vms = [example[5] for example in trainset]
    else:
        input_ids = torch.LongTensor([example[0] for example in trainset])
        label_ids = torch.LongTensor([example[1] for example in trainset])
        mask_ids = torch.LongTensor([example[2] for example in trainset])
        pos_ids = torch.LongTensor([example[3] for example in trainset])
        vms = [example[4] for example in trainset]
        postag_ids = None

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    if not os.path.exists("./log"):
        os.mkdir("./log")

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    total_loss = 0.
    result = 0.0
    best_result = 0.0

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        for i, (
                input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch,
                vms_batch) in enumerate(
            batch_loader(batch_size, input_ids, postag_ids, label_ids, mask_ids, pos_ids, vms)):
            model.zero_grad()

            vms_batch = torch.LongTensor(vms_batch)

            input_ids_batch = input_ids_batch.to(device)
            postag_ids_batch = postag_ids_batch.to(device) if args.use_postag else None
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            loss, logits = model(input_ids_batch, postag_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch,
                                 vm=vms_batch)
            preds = torch.argmax(logits, dim=-1)
            # print(f"preds: {preds}")
            # print(f"label_ids_batch: {label_ids_batch}")
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                  total_loss / args.report_steps))
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

        print("Start evaluation on dev dataset.")
        result = evaluate(args, False)
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
        else:
            continue
        # only test on test dataset when the model train finished.
        # print("Start evaluation on test dataset.")
        # evaluate(args, True)

    # Evaluation phase.
    print("Final evaluation on the test dataset.")

    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))
    evaluate(args, True)


if __name__ == "__main__":
    main()
