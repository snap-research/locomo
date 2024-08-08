from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
import random
import os, json, sys
from tqdm import tqdm
import time
import argparse
import torch
from transformers import pipeline


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True, type=str)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--ref-dir', default='', type=str)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--openai-key-file', type=str, default='')
    parser.add_argument('--use-rag', action="store_true")
    args = parser.parse_args()
    return args


def initimacy(args):

    device = torch.device('cuda')
    model_name = "cardiffnlp/twitter-roberta-large-intimacy-latest"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # transfer model
    # model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # with pipeline
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

    data_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    if len(data_files) > 10:
        data_files = random.sample(data_files, k=10)
    
    for d in data_files:
        data = json.load(open(os.path.join(args.data_dir, d)))
        for k in data.keys():
            if not k.startswith('session') or 'date_time' in k:
                continue
            for j in tqdm(range(len(data[k])), desc='Iterating over %s' % k):
                try:
                    res = pipe(data[k][j]['clean_text'])
                except:
                    res = pipe(data[k][j]['text'])
                score = res[0]['score']
                data[k][j]['intimacy_score'] = score
        with open(os.path.join(args.out_dir, d), 'w') as f:
            json.dump(data, f, indent=2)


def initimacy_question(args):

    device = torch.device('cuda')
    question_model_name = "shahrukhx01/question-vs-statement-classifier"
    question_model = AutoModelForSequenceClassification.from_pretrained(question_model_name)
    question_tokenizer = AutoTokenizer.from_pretrained(question_model_name)
    model_name = "pedropei/question-intimacy"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # transfer model
    # model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # with pipeline
    qpipe = pipeline("text-classification", model=question_model, tokenizer=question_tokenizer, device=0)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)

    data_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    if len(data_files) > 10:
        data_files = random.sample(data_files, k=10)
    
    count = 0
    total = 0
    for d in data_files:
        data = json.load(open(os.path.join(args.data_dir, d)))
        for k in data.keys():
            if not k.startswith('session') or 'date_time' in k:
                continue
            for j in tqdm(range(len(data[k])), desc='Iterating over %s' % k):
                
                total += 1
                # classify question
                try:
                    res = qpipe(data[k][j]['clean_text'])
                except:
                    res = qpipe(data[k][j]['text'])
                label = res[0]['label']
                if label != 'LABEL_1':
                    continue
            
                try:
                    res = pipe(data[k][j]['clean_text'])
                except:
                    res = pipe(data[k][j]['text'])
                score = res[0]['score']
                data[k][j]['intimacy_question_score'] = score
                count += 1
        with open(os.path.join(args.out_dir, d), 'w') as f:
            json.dump(data, f, indent=2)
    print("Found %s questions in %s dialogs" % (count, total))

def emotion(args):

    device = torch.device('cuda')
    model_name = "cardiffnlp/twitter-roberta-large-emotion-latest"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # transfer model
    # model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # with pipeline
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)

    if args.ref_dir != '':
        data_files = [f for f in os.listdir(args.ref_dir) if f.endswith('.json')]
    else:
        data_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]

    
    if len(data_files) > 10:
        data_files = random.sample(data_files, k=10)

    for d in data_files:

        data = json.load(open(os.path.join(args.data_dir, d)))
        if os.path.exists(os.path.join(args.out_dir, d)):
            data = json.load(open(os.path.join(args.out_dir, d)))

        for k in data.keys():
            if not k.startswith('session') or 'date_time' in k:
                continue
            for j in tqdm(range(len(data[k])), desc='Iterating over %s' % k):
                try:
                    res = pipe(data[k][j]['text'])[0]
                except:
                    res = pipe(data[k][j]['clean_text'])[0]
                # print(res)
                predictions = [x for x in res]
                data[k][j]['emotion'] = predictions
        
        with open(os.path.join(args.out_dir, d), 'w') as f:
            json.dump(data, f, indent=2)


def sentiment(args):

    device = torch.device('cuda')
    model_name = "cardiffnlp/twitter-roberta-base-2021-124m-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    # transfer model
    # model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # with pipeline
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)

    if args.ref_dir != '':
        data_files = [f for f in os.listdir(args.ref_dir) if f.endswith('.json')]
    else:
        data_files = [f for f in os.listdir(args.data_dir) if f.endswith('.json')]
    
    if len(data_files) > 10:
        data_files = random.sample(data_files, k=10)

    for d in data_files:

        data = json.load(open(os.path.join(args.data_dir, d)))
        if os.path.exists(os.path.join(args.out_dir, d)):
            data = json.load(open(os.path.join(args.out_dir, d)))

        for k in data.keys():
            if not k.startswith('session') or 'date_time' in k:
                continue
            for j in tqdm(range(len(data[k])), desc='Iterating over %s' % k):
                try:
                    res = pipe(data[k][j]['text'])[0]
                except:
                    res = pipe(data[k][j]['clean_text'])[0]

                if 'sentiment' in data[k][j]:
                    data[k][j]['emotion'] = data[k][j]['sentiment'].copy()
                
                data[k][j]['sentiment'] = res

        with open(os.path.join(args.out_dir, d), 'w') as f:
            json.dump(data, f, indent=2)


def main():

    args = parse_args()
    if args.mode == 'intimacy':
        initimacy(args)
    elif args.mode == 'intimacy_question':
        initimacy_question(args)
    elif args.mode == 'emotion':
        emotion(args)
    elif args.mode == 'sentiment':
        sentiment(args)
    else:
        raise ValueError


if __name__=="__main__":
    main()