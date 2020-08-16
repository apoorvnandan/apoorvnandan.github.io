---
layout: post
comments: true
title:  "BERT masked LM training"
excerpt: "Pretraining or fine tuning BERT on masked LM task"
date:   2020-08-15 22:00:00
---

## Initial Setup
I will use BERT model from huggingface and a lighweight wrapper over pytorch
called [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to avoid writing boilerplate.<br/>
```bash
!pip install transformers
!pip install pytorch-lightning
```
To run this over TPUs, the following dependencies are also needed.<br/>
```bash
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly  --apt-packages libomp5 libopenblas-dev
```
To demonstrate I'll use a text corpus, which can be downloaded as follows:<br/>
```python

import urllib.request
txt_url = "https://s3.amazonaws.com/text-datasets/nietzsche.txt"
urllib.request.urlretrieve(txt_url, 'train.txt')

```
## Imports and Configs<br/>
```python

import pytorch_lightning as pl
from argparse import Namespace
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    AdamW,
    DataCollatorForLanguageModeling
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

args = Namespace()
args.train = "train.txt"
args.max_len = 128
args.model_name = "bert-base-uncased"
args.epochs = 1
args.batch_size = 4

```
## Create Dataloader<br/>
The Dataset class reads a text file.
Each line in the file forms a single element of the dataset after tokenization with BERT's tokenizer.
```python

tokenizer = BertTokenizer.from_pretrained(args.model_name)

class MaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer):
        self.tokenizer = tokenizer
        self.lines = self.load_lines(file)
        self.ids = self.encode_lines(self.lines)
        
    def load_lines(self, file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines
    
    def encode_lines(self, lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=args.max_len
        )
        return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)
        
train_dataset = MaskedLMDataset(args.train, tokenizer)
```
A collator function in pytorch takes a list of elements given by the dataset class and
and creates a batch of input (and targets). Huggingface provides a convenient collator function
which takes a list of input ids from my dataset, masks 15% of the tokens, and
creates a batch after appropriate padding.

Targets are created by cloning the input ids. Then, if a token is supposed to be masked, the corresponding input id is replaced 
by that of either the [MASK] token (80% chance), a random token (10% chance), the same token (10% chance).
If a token is not supposed to be masked, the corresponding target id is replaced by -100, so that they are ignored during loss calculation.
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.bs,
    collate_fn=data_collator
)

```
## Define model, training step and optmizer<br/>
```python
class Bert(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained(args.model_name)

    def forward(self, input_ids, labels):
        return self.bert(input_ids=input_ids,labels=labels)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        outputs = self(input_ids=input_ids, labels=labels)
        loss = outputs[0]
        return {"loss": loss}

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-5)

model = Bert()

```
## Train<br/>
This is where pytorch lightning does an awesome job. Once the model and
data loader are ready, I can train on CPU, single GPU, multiple GPUs, single TPU core and multiple TPU cores with just two lines of code.<br>
1. Initialise the Trainer as per the hardware:<br>
    CPU<br>
    ```python
    trainer = pl.Trainer(max_epochs=1)
    ```
    GPU (single or multiple)<br>
    ```python
    trainer = pl.Trainer(max_epochs=1, gpus=8)
    ```
    Single TPU core<br>
    ```python
    trainer = pl.Trainer(max_epochs=1, tpu_cores=[1])
    ```
    Multiple TPU cores<br>
    ```python
    trainer = pl.Trainer(max_epochs=1, tpu_cores=8)
    ```
2. Run the fit function.
```python
trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader)
```
## Saving and Loading
The weights can be saved and loaded for predictions like this.<br>
```python
torch.save(model.state_dict(), 'saved.bin')

class BertPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForMaskedLM.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, labels=None):
        return self.bert(input_ids=input_ids,labels=labels)

new_model = BERTPred()
new_model.load_state_dict(torch.load('saved.bin'))
new_model.eval()
```
