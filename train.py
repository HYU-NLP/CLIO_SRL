from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
import torch
import pandas as pd
import random
import numpy as np
import json
import evaluate
import argparse

class SRL_Dataset(Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def BIO_tagging_y(total_data, tokenizer):
    # BIO tagging
    train_data = []

    for i in total_data:
        sent = i['line']
        lemma = i['predicate']['lemma']

        # add <p>
        temp = sent.split()
        if i['predicate']['word_id'] is None:
            continue
        p_id_s = i['predicate']['word_id'][0]
        p_id_e = i['predicate']['word_id'][-1]
        pred = ' '.join(temp[p_id_s:p_id_e+1])
        split_sent = temp[:p_id_s] + ['<p>'] + temp[p_id_s:p_id_e+1] + ['</p>'] + temp[p_id_e+1:]

        input_s = ' '.join(split_sent)
        token_s = tokenizer.tokenize(input_s)
        visited = [0] * len(token_s)
        label = token_s[:]

        # Arg 0
        if i['arg0']['word']:
            a_fidx = -1
            arg_tokenized = tokenizer.tokenize(i['arg0']['word'])
            try:
                for k, arg in enumerate(arg_tokenized):
                    idx = label.index(arg)   
                    
                    while k and idx != a_fidx+1:
                        idx = label[idx+1:].index(arg) + idx + 1
                    
                    if k:
                        label[idx] = 'I-' + 'ARG0'
                        visited[idx] = 1
                        a_fidx = idx
                    else:
                        label[idx] = 'B-' + 'ARG0'
                        visited[idx] = 1
                        a_fidx = idx
            except:
                continue
        
        # ARG 1
        if i['arg1']['word']:
            a_fidx = -1
            arg_tokenized = tokenizer.tokenize(i['arg1']['word'])
            try:
                for k, arg in enumerate(arg_tokenized):
                    idx = label.index(arg)   
                    
                    while k and idx != a_fidx+1:
                        idx = label[idx+1:].index(arg) + idx + 1
                    
                    if k:
                        label[idx] = 'I-' + 'ARG1'
                        visited[idx] = 1
                        a_fidx = idx
                    else:
                        label[idx] = 'B-' + 'ARG1'
                        visited[idx] = 1
                        a_fidx = idx
            except:
                continue
        
        # O tagging
        for j, k in enumerate(visited):
            if k:
                continue
            else:
                label[j] = 'O'

        train_data.append({'input': input_s, 'label': label, 'predicate': pred, 'lemma': lemma})
    print("Total Data: ", len(train_data))
    return pd.DataFrame(train_data)

def tokenize_label(df, max_len):
    all_labels = ['O','B-ARG0', 'B-ARG1']
    
    dict = {string : i for i,string in enumerate(all_labels)}
    total_labels = []

    for label in df['label']:
        labels = [-100]
        for i in label:
            if i == 'X':
                labels.append(-100)
            else:
                labels.append(dict[i])
        labels.append(-100)
        while len(labels) < max_len:
            labels.append(-100)
        total_labels.append(labels)
    return total_labels


def tokenized_input(df):
    sentence = []
    for i in df['input']:
        sentence.append(i)
    return tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=256)



def compute_metrics(eval_preds):
    label_names = [ 'O','B-ARG0', 'B-ARG1']
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": all_metrics["overall_precision"]*100,
        "recall": all_metrics["overall_recall"]*100,
        "f1": all_metrics["overall_f1"]*100,
        "accuracy": all_metrics["overall_accuracy"]*100,
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="klue/bert-base", help="model to train (default: klue/bert-base)")
    parser.add_argument("--output_dir", type=str, default="./results", help="directory which stores various outputs (default: ./results)")
    parser.add_argument("--save_steps", type=int, default=500, help="interval of saving model (default: 500)")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="number of train epochs (default: 20)")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate (default: 5e-5)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help=" (default: 16)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16, help=" (default: 16)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help=" (default: 0.01)")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help=" (default: steps)")
    parser.add_argument("--eval_steps", type=int, default=500, help=" (default: 500)")
    parser.add_argument("--save_pretrained", type=str, default="./best_model", help=" (default: ./best_model)")
    parser.add_argument('--train_file_path', type=str, default="./data/clio_train.json")
    parser.add_argument('--val_file_path', type=str, default="./data/clio_val.json")
    parser.add_argument('--test_file_path', type=str, default="./data/clio_test.json")
    args = parser.parse_args()

    seed_everything(42)


    label_names = [ 'O','B-ARG0', 'B-ARG1']
    metric = evaluate.load('seqeval')
    model_checkpoint = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, additional_special_tokens=['<p>', '</p>'])#, force_download=True)

    train_file_path = args.train_file_path
    eval_file_path = args.val_file_path

    train_data = json.load(open(train_file_path))
    eval_data = json.load(open(eval_file_path))

    train = BIO_tagging_y(train_data, tokenizer)
    eval = BIO_tagging_y(eval_data, tokenizer)

    tokenized_train_input = tokenized_input(train)
    tokenized_train_label = tokenize_label(train, tokenized_train_input['input_ids'].size()[1])
    tokenized_val_input = tokenized_input(eval)
    tokenized_val_label = tokenize_label(eval, tokenized_val_input['input_ids'].size()[1])

    print("*************tokenized finish****************")

    SRL_train_dataset = SRL_Dataset(tokenized_train_input, tokenized_train_label)
    SRL_val_dataset = SRL_Dataset(tokenized_val_input, tokenized_val_label)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    model_config =  AutoConfig.from_pretrained(model_checkpoint)
    model_config.num_labels = 3 # all label

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, config=model_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)


    training_args = TrainingArguments(
        output_dir=args.output_dir,                     # output directory
        save_total_limit=5,
        num_train_epochs=args.num_train_epochs,         # total number of training epochs
        learning_rate=args.learning_rate,               # learning rate
        per_device_train_batch_size=args.per_device_train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # batch size for evaluation
        weight_decay=args.weight_decay,                # strength of weight decay
        eval_strategy=args.evaluation_strategy,  # evaluation strategy to adopt during training
                                                    # `no`: No evaluation during training.
                                                    # `steps`: Evaluate every `eval_steps`.
                                                    # `epoch`: Evaluate every end of epoch.
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        save_strategy="epoch",
        fp16=True
    )
    trainer = Trainer(
      model=model,                         # the instantiated 🤗 Transformers model to be trained
      args=training_args,                  # training arguments, defined above
      train_dataset=SRL_train_dataset,         # training dataset
      eval_dataset=SRL_val_dataset,            # evaluation dataset
      compute_metrics=compute_metrics,         # define metrics function
    )


    trainer.train()
    model.save_pretrained(args.save_pretrained)

    test_file_path = args.test_file_path
    test_data = json.load(open(test_file_path))
    test = BIO_tagging_y(test_data,tokenizer)
    tokenized_test_input = tokenized_input(test)
    tokenized_test_label = tokenize_label(test, tokenized_train_input['input_ids'].size()[1])
    SRL_test_dataset = SRL_Dataset(tokenized_test_input, tokenized_test_label)

    trainer.evaluate(
        eval_dataset = SRL_test_dataset
    )
