#!/usr/bin/env python

import os
import sys
import numpy as np

from argparse import ArgumentParser

from datasets import Dataset, DatasetDict
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer
)

DEFAULTS = {
    'DATA_DIR': 'data',
    'TOKENIZER': 'tokenizer',
    'MODEL': 'model',
    'MAX_LENGTH': 512,
    'LEARNING_RATE': 1e-5,
    'BATCH_SIZE': 16,
    'EPOCHS': 4,
}

def argparser():
    ap = ArgumentParser()
    ap.add_argument(
        '--tokenizer',
        metavar='DIR',
        default=DEFAULTS['TOKENIZER']
    )
    ap.add_argument(
        '--model',
        metavar='DIR',
        default=DEFAULTS['MODEL']
    )
    ap.add_argument(
        '--data',
        metavar='DIR',
        default=DEFAULTS['DATA_DIR']
    )
    ap.add_argument(
        '--max_length',
        type=int,
        default=DEFAULTS['MAX_LENGTH']
    )
    ap.add_argument(
        '--batch_size',
        type=int,
        default=DEFAULTS['BATCH_SIZE']
    )
    ap.add_argument(
        '--epochs',
        type=int,
        default=DEFAULTS['EPOCHS']
    )
    ap.add_argument(
        '--learning_rate',
        type=float,
        default=DEFAULTS['LEARNING_RATE']
    )
    return ap


def load_data(fn):
    labels, texts = [], []
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            l = l.rstrip('\n')
            label, text = l.split(None, 1)
            label = label.replace('__label__', '')
            labels.append(label)
            texts.append(text)
    dataset = Dataset.from_dict({
        'label_str': labels,
        'text': texts,
    })
    return dataset


def load_datasets(directory):
    datasets = DatasetDict()
    for s in ('train', 'dev', 'test'):
        datasets[s] = load_data(os.path.join(directory, f'{s}.txt'))
    return datasets


def load_model(directory, num_labels, args):
    model = GPT2ForSequenceClassification.from_pretrained(
        directory,
        num_labels=num_labels
    )
    model.config.max_length = args.max_length
    return model


def load_tokenizer(directory, args):
    tokenizer = GPT2Tokenizer.from_pretrained(directory)
    tokenizer.add_special_tokens({ 
        "eos_token": "</s>",
        "bos_token": "<s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    })
    tokenizer.add_prefix_space = True
    tokenizer.model_max_length = args.max_length
    return tokenizer


def make_encode_text_function(tokenizer):
    def encode_text(example):
        encoded = tokenizer(
            example['text'],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            #return_tensors='pt',
        )
        # Adding return_tensorts='pt' to the tokenizer call results in
        # the tokenizer adding an unnecessary additional batch dimension,
        # necessitating the lines below to remove it
        # encoded['input_ids'] = encoded['input_ids'][0]
        # encoded['attention_mask'] = encoded['attention_mask'][0]
        return encoded
    return encode_text


def make_encode_label_function(labels):
    label_map = { l: i for i, l in enumerate(labels) }
    def encode_label(example):
        example['label'] = label_map[example['label_str']]
        return example
    return encode_label


def accuracy(pred):
    y_pred = pred.predictions.argmax(axis=1)
    y_true = pred.label_ids
    return { 'accuracy': sum(y_pred == y_true) / len(y_true) }


def main(argv):
    args = argparser().parse_args(argv[1:])

    data = load_datasets(args.data)
    labels = list(set(l for d in data.values() for l in d['label_str']))

    tokenizer = load_tokenizer(args.tokenizer, args)
    
    encode_text = make_encode_text_function(tokenizer)
    encode_label = make_encode_label_function(labels)
    
    data = data.map(encode_text)
    data = data.map(encode_label)
    
    model = load_model(args.model, len(labels), args)
    
    # This needs to be set explicitly for some reason
    model.config.pad_token_id = tokenizer.pad_token_id
    
    train_args = TrainingArguments(
        'output_dir',
        save_strategy='no',
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=data['train'],
        eval_dataset=data['dev'],
        tokenizer=tokenizer,
        compute_metrics=accuracy
    )

    trainer.train()

    print(trainer.evaluate(data['test'], metric_key_prefix='test'))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
