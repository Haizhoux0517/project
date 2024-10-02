import argparse
import os

import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn, optim
from transformers import DistilBertTokenizer, DistilBertModel, get_linear_schedule_with_warmup, AutoModelForCausalLM
from torch.optim.adamw import AdamW
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformerModel(nn.Module):

    def __init__(self):

        super(TransformerModel, self).__init__()
        torch.manual_seed(1)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classification_head = nn.Sequential(
            nn.Linear(768, 50),
            nn.ReLU(),
            torch.nn.Dropout(0.1),
            nn.Linear(50, 2),

        )
        self.optimizers = [optim.Adam(list(self.bert.parameters()), lr=5e-5),
                               optim.SGD(list(self.classification_head.parameters()), lr=05e-5)]

    def forward(self, input_ids, attention_mask):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        last_hidden_state = outputs[0][:, 0, :]
        logits = self.classification_head(last_hidden_state)
        return logits

def evaluate_roc(probs, y_true):

    preds = probs[:, 1]

    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f_measure = f1_score(y_true, y_pred)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F-measure: {f_measure:.4f}')
    values = [accuracy, precision, recall, f_measure]
    plt.bar([1, 2, 3, 4], values, tick_label=["Accuracy", "Precision", "Recall", "F-measure"])

    plt.title("Accuracy, Precision, Recall, F-measure")
    plt.xlabel("Metrics", fontweight='bold', fontsize='12', horizontalalignment='center')
    plt.ylabel("Values",  fontweight='bold', fontsize='12')
    plt.show()


def train_validation(train_csv):
    X_toxic = train_csv[train_csv['toxic'] == 1]['comment_text'].values
    X_non_toxic = train_csv[train_csv['toxic'] == 0]['comment_text'].sample(15294).values

    X_ST = train_csv[train_csv['severe_toxic'] == 1]['comment_text'].values
    X_ST_non = train_csv[train_csv['severe_toxic'] == 0]['comment_text'].sample(15294).values

    X_O_toxic = train_csv[train_csv['obscene'] == 1]['comment_text'].values
    X_O_non= train_csv[train_csv['obscene'] == 0]['comment_text'].sample(15294).values

    X_Th_ = train_csv[train_csv['threat'] == 1]['comment_text'].values
    X_Th_non = train_csv[train_csv['threat'] == 0]['comment_text'].sample(15294).values

    X_I= train_csv[train_csv['insult'] == 1]['comment_text'].values
    X_I_non = train_csv[train_csv['insult'] == 0]['comment_text'].sample(15294).values

    X_Ih = train_csv[train_csv['identity_hate'] == 1]['comment_text'].values
    X_Ih_non = train_csv[train_csv['identity_hate'] == 0]['comment_text'].sample(15294).values

    X = np.concatenate([X_toxic, X_non_toxic, X_ST, X_ST_non, X_O_toxic, X_O_non, X_Th_, X_Th_non, X_I, X_I_non, X_Ih, X_Ih_non])

    Y_toxic = train_csv[train_csv['toxic'] == 1]['toxic'].values
    Y_non_toxic = train_csv[train_csv['toxic'] == 0]['toxic'].sample(15294).values

    Y_ST = train_csv[train_csv['severe_toxic'] == 1]['severe_toxic'].values
    Y_ST_non = train_csv[train_csv['severe_toxic'] == 0]['severe_toxic'].sample(15294).values

    Y_O_toxic = train_csv[train_csv['obscene'] == 1]['obscene'].values
    Y_O_non= train_csv[train_csv['obscene'] == 0]['obscene'].sample(15294).values

    Y_Th_ = train_csv[train_csv['threat'] == 1]['threat'].values
    Y_Th_non = train_csv[train_csv['threat'] == 0]['threat'].sample(15294).values

    Y_I= train_csv[train_csv['insult'] == 1]['insult'].values
    Y_I_non = train_csv[train_csv['insult'] == 0]['insult'].sample(15294).values

    Y_Ih = train_csv[train_csv['identity_hate'] == 1]['identity_hate'].values
    Y_Ih_non = train_csv[train_csv['identity_hate'] == 0]['identity_hate'].sample(15294).values

    Y = np.concatenate([Y_toxic, Y_non_toxic, Y_ST, Y_ST_non, Y_O_toxic, Y_O_non, Y_Th_, Y_Th_non, Y_I, Y_I_non, Y_Ih, Y_Ih_non])

    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.5, random_state=42)
    return X_train, X_val, y_train, y_val

def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')


    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent.strip(),
            add_special_tokens=True,
            max_length=200,
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


loss_fn = nn.CrossEntropyLoss()


def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):

    print("Start training...\n")

    for epoch in range(epochs):
        print(f"{'Epoch':^5}  || {'Batch':^5} || {'Train Loss':^10} || {'Val Loss':^8} || {'Val Acc':^7}")
        print("\n")
        step_loss, step_count = 0, 0
        total_loss = 0
        model.train()

        for i, batch in enumerate(train_dataloader):

            input_ids, attn_mask, labels = tuple(t.to(device) for t in batch)

            model.zero_grad()

            logits = model(input_ids, attn_mask)

            loss = loss_fn(logits, labels)
            step_loss += loss.item()
            total_loss += loss.item()
            step_count += 1
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            print(f"{epoch + 1:^5}  || {i:^5} || {step_loss / step_count:^10.6f}")
            step_loss, step_count = 0, 0

        train_loss = total_loss / len(train_dataloader)

        if evaluation == True:

            val_loss, val_accuracy = evaluate(model, val_dataloader)
            print(
            f"{epoch + 1:^5}  || {'':^5} || {train_loss:^10.6f} || {val_loss:^8.6f} || {val_accuracy:^7.6f} ")
            print("\n")
    print("Training complete!")

def evaluate(model, val_dataloader):

    model.eval()

    val_accuracy = []
    val_loss = []

    for batch in val_dataloader:
        input_ids, attn_mask, labels = tuple(t.to(device) for t in batch)


        with torch.no_grad():
            logits = model(input_ids, attn_mask)

        loss = loss_fn(logits, labels)
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy

def distilbert_predict(model, test_dataloader):

    model.eval()

    all_logits = []

    for batch in test_dataloader:

        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


def initialize_model(epochs=4):

    # Instantiate Bert Classifier
    distilbert_classifier = TransformerModel()

    distilbert_classifier.to(device)

    optimizer = AdamW(distilbert_classifier.parameters(),
                      lr=5e-5,
                      eps=1e-8
                      )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return distilbert_classifier, optimizer, scheduler

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", dest="inputfile",
                           default=os.path.join('data', 'input', 'dev.txt'))
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=1)
    argparser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=32)
    argparser.add_argument("-M", "--basemodel", dest="basemodel",
                           default='distilbert-base-uncased')
    opts = argparser.parse_args()

    train_csv = pd.read_csv('data/jigsaw-toxic-comment-classification-challenge/train.csv')
    x_train, x_val, y_train, y_val = train_validation(train_csv)

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    train_inputs, train_masks = preprocessing_for_bert(x_train)
    val_inputs, val_masks = preprocessing_for_bert(x_val)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=opts.batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=opts.batch_size)
    distilbert_classifier, optimizer, scheduler = initialize_model(epochs=4)
    train(distilbert_classifier, train_dataloader, val_dataloader, epochs=opts.epochs, evaluation=True)

    probs = distilbert_predict(distilbert_classifier, val_dataloader)
    evaluate_roc(probs, y_val)