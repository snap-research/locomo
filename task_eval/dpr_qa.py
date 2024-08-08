import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import os, json
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from global_methods  import get_embedding, set_openai_key, run_chatgpt_with_examples
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_acc

# Just load seaborn & set theme and the chart looks better:
import seaborn as sns
sns.set_theme()

def get_conversation_facts():

    CONVERSATION2FACTS_PROMPT = """
    Convert the given CONVERSATION into a list of FACTS about each speaker.\n\n
    """

    # Step 1: get events
    task = json.load(open(os.path.join(args.prompt_dir, 'fact_generation_examples.json')))
    query = CONVERSATION2FACTS_PROMPT
    examples = [[task['input_prefix'] + e["input"], json.dumps(e["output"], indent=2)] for e in task['examples']]

    conversation = ""
    conversation += agent_a['session_%s_date_time' % session_idx] + '\n'
    for dialog in agent_a['session_%s' % session_idx]:
        # if 'clean_text' in dialog:
        #     writer.write(dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n')
        # else:
        conversation += dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"\n'
    
    input = task['input_prefix'] + conversation
    try:
        output = run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1500, use_16k=False).strip()
        print(output)
        facts = json.loads(output)
    except json.decoder.JSONDecodeError:
        output = run_chatgpt_with_examples(query, examples, input, num_gen=1, num_tokens_request=1500, use_16k=False).strip()
        facts = json.loads(output)


def save_eval(data_file, accs, key='exact_match'):

    
    if os.path.exists(data_file.replace('.json', '_scores.json')):
        data = json.load(open(data_file.replace('.json', '_scores.json')))
    else:
        data = json.load(open(data_file))

    assert len(data['qa']) == len(accs), (len(data['qa']), len(accs), accs)
    for i in range(0, len(data['qa'])):
        data['qa'][i][key] = accs[i]
    
    with open(data_file.replace('.json', '_scores.json'), 'w') as f:
        json.dump(data, f, indent=2)


# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True, type=str)
    parser.add_argument('--retriever', required=True, type=str)
    parser.add_argument('--reader', required=False, type=str)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--openai-key-file', type=str, default='')
    parser.add_argument('--use-16k', action="store_true")
    args = parser.parse_args()
    return args


def init_context_model(args):

    if args.retriever == 'dpr':
        from transformers import DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        context_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").cuda()
        context_model.eval()
        return context_tokenizer, context_model

    elif args.retriever == 'contriever':

        from transformers import AutoTokenizer, AutoModel
        context_tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        context_model = AutoModel.from_pretrained('facebook/contriever').cuda()
        context_model.eval()
        return context_tokenizer, context_model

    elif args.retriever == 'dragon':

        from transformers import AutoTokenizer, AutoModel
        context_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        context_model = AutoModel.from_pretrained('facebook/dragon-plus-context-encoder').cuda()
        return context_tokenizer, context_model

    elif args.retriever == 'openai':

        set_openai_key(args)
        return None, None
    
    else:
        raise ValueError
    
def init_query_model(args):

    if args.retriever == 'dpr':
        from transformers import DPRConfig, DPRContextEncoder, DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        question_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base").cuda()
        question_model.eval()
        return question_tokenizer, question_model

    elif args.retriever == 'contriever':

        from transformers import AutoTokenizer, AutoModel
        question_tokenizer = context_tokenizer
        question_model = AutoModel.from_pretrained('facebook/contriever').cuda()
        question_model.eval()
        return question_tokenizer, question_model

    elif args.retriever == 'dragon':

        from transformers import AutoTokenizer, AutoModel
        context_tokenizer = AutoTokenizer.from_pretrained('facebook/dragon-plus-query-encoder')
        question_model = AutoModel.from_pretrained('facebook/dragon-plus-query-encoder').cuda()
        question_tokenizer = context_tokenizer
        return question_tokenizer, question_model

    elif args.retriever == 'openai':

        set_openai_key(args)
        return None, None
    
    else:
        raise ValueError


def get_embeddings(args, inputs, mode='context'):

    if mode == 'context':
        tokenizer, encoder = init_context_model(args)
    else:
        tokenizer, encoder = init_query_model(args)
    
    all_embeddings = []
    batch_size = 24
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs), batch_size)):
            # print(input_ids.shape)
            if args.retriever == 'dpr':
                input_ids = tokenizer(inputs[i:(i+batch_size)], return_tensors="pt", padding=True)["input_ids"].cuda()
                embeddings = encoder(input_ids).pooler_output.detach()
                # print(embeddings.shape)
                all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
            elif args.retriever == 'contriever':
                # Compute token embeddings
                ctx_input = tokenizer(inputs[i:(i+batch_size)], padding=True, truncation=True, return_tensors='pt')
                # print(ctx_input.keys())
                # input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
                outputs = encoder(**ctx_input)
                embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
                all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
            elif args.retriever == 'dragon':
                ctx_input = tokenizer(inputs[i:(i+batch_size)], padding=True, truncation=True, return_tensors='pt').to(device)
                embeddings = encoder(**ctx_input).last_hidden_state[:, 0, :]
                # all_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                all_embeddings.append(embeddings)
            elif args.retriever == 'openai':
                all_embeddings.append(torch.tensor(get_embedding(inputs)))
            else:
                raise ValueError

    return torch.cat(all_embeddings, dim=0).cpu().numpy()


def get_context_embeddings(args, data, context_tokenizer, context_encoder, captions=None):

    context_embeddings = []
    context_ids = []
    for i in tqdm(range(1,20), desc="Getting context encodings"):
        contexts = []
        if 'session_%s' % i in data:
            date_time_string = data['session_%s_date_time' % i]
            for dialog in data['session_%s' % i]:

                turn = ''
                # conv = conv + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                try:
                    turn = dialog['speaker'] + ' said, \"' + dialog['compressed_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['compressed_text'] + '\n'
                except KeyError:
                    turn = dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'
                if "img_file" in dialog and len(dialog["img_file"]) > 0:
                    turn += '[shares %s]\n' % dialog["blip_caption"]
                contexts.append('(' + date_time_string + ') ' + turn)

                # if 'caption' in dialog and captions is not None:
                #     try:
                #         caption = captions['session_%s/a/%s' % (i, dialog["img_file"][0])]
                #         contexts.append('(' + date_time_string + ') ' + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\" and shared ' + caption + '\n')
                #     except:
                #         contexts.append('(' + date_time_string + ') ' + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n')
                # else:
                #     contexts.append('(' + date_time_string + ') ' + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n')
                
                # contexts.append('(' + date_time_string + ') ' + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n')


                context_ids.append(dialog["dia_id"])
            with torch.no_grad():
                # print(input_ids.shape)
                if args.retriever == 'dpr':
                    input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
                    embeddings = context_encoder(input_ids).pooler_output.detach()
                    # print(embeddings.shape)
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif args.retriever == 'contriever':
                    # Compute token embeddings
                    inputs = context_tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
                    print(inputs.keys())
                    # input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
                    outputs = context_encoder(**inputs)
                    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif args.retriever == 'dragon':
                    ctx_input = context_tokenizer(contexts, padding=True, truncation=True, return_tensors='pt')
                    embeddings = context_encoder(**ctx_input).last_hidden_state[:, 0, :]
                    context_embeddings.append(torch.nn.functional.normalize(embeddings, dim=-1))
                elif args.retriever == 'openai':
                    context_embeddings.append(torch.tensor(get_embedding(contexts)))
                else:
                    raise ValueError

    # print(context_embeddings[0].shape[0])
    context_embeddings = torch.cat(context_embeddings, dim=0)
    # print(context_embeddings.shape[0])

    return context_ids, context_embeddings


def analyze_recall(data_file):

    total_counts = defaultdict(lambda: 0)
    hard_memory_counts = defaultdict(lambda: 0)
    easy_memory_counts = defaultdict(lambda: 0)
    recall_counts = {}
    hard_memory_recall = {}
    easy_memory_recall = {}
    for top_k in [1, 5, 10, 15, 20, 25, 50, 75, 100, 150]:
        recall_counts[top_k] = defaultdict(lambda: 0)
        hard_memory_recall[top_k] = defaultdict(lambda: 0)
        easy_memory_recall[top_k] = defaultdict(lambda: 0)

    data = json.load(open(data_file))
    for i, qa in tqdm(enumerate(data['qa'])):
        total_counts[qa['category']] += 1

        if 'context_match' in qa:
            for top_k in [75, 100, 150]:
                if all([ev in qa["context_match"][:top_k] for ev in qa["evidence"]]):
                    recall_counts[top_k][qa['category']] += 1
                for num, ev in enumerate(qa["evidence"]):
                    session_num = int(ev.split(':')[0][1:])
                    if qa['category'] in [1, 5]:
                        if top_k == 1:
                            easy_memory_counts[session_num] += 1
                        if ev in qa["context_match"][:top_k]:
                            easy_memory_recall[top_k][session_num] += 1
                        else:
                            print(top_k, qa['question'], qa['answer'])
                    else:
                        if top_k == 1:
                            hard_memory_counts[session_num] += 1
                        if ev in qa["context_match"][:top_k]:
                            hard_memory_recall[top_k][session_num] += 1
                            print(top_k, qa['question'], qa['answer'])


def run_recall_eval(args, data, out_file, context_embeddings, question_model, question_tokenizer, dialog_ids):

    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    for i, qa in tqdm(enumerate(data['qa'])):

        # if qa['category'] == 3:
        #     question = 'Today is 30 May, 2020.' + qa['question'] + ' Give an approximate date, month or year.'
        # else:
        question = qa['question']

        if args.retriever == 'dpr':
            input_ids = question_tokenizer(question, return_tensors="pt")["input_ids"].cuda()
            question_embedding = torch.nn.functional.normalize(question_model(input_ids).pooler_output.detach(), dim=-1)

        elif args.retriever == 'contriever':
            # Compute token embeddings
            inputs = question_tokenizer(question, padding=True, truncation=True, return_tensors='pt')
            print(inputs.keys())
            # input_ids = context_tokenizer(contexts, return_tensors="pt", padding=True)["input_ids"].cuda()
            outputs = question_model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            question_embedding = torch.nn.functional.normalize(embeddings, dim=-1)

        elif args.retriever == 'dragon':
            query_input = question_tokenizer(question, return_tensors='pt')
            question_embedding = question_model(**query_input).last_hidden_state[:, 0, :]
            question_embedding = torch.nn.functional.normalize(question_embedding, dim=-1)

        elif args.retriever == 'openai':
            question_embedding = torch.tensor(get_embedding(question))

        with torch.no_grad():
            output = cos(context_embeddings, question_embedding).squeeze().cpu().numpy()
        sorted_outputs = np.argsort(output)[::-1]
        sorted_dia_ids = [dialog_ids[idx] for idx in sorted_outputs]
        
        data['qa'][i]['context_match'] = sorted_dia_ids

    with open(out_file, 'w') as f:
        json.dump(data, f, indent=2)


def plot_recall_by_category(data_files, labels):

    for f_num, data_file in enumerate(data_files):

        total_counts = defaultdict(lambda: 0)
        recall_counts = {}
        for top_k in [1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200]:
            recall_counts[top_k] = defaultdict(lambda: 0)

        data = json.load(open(data_file))
        total_context_length = 0
        for i, qa in tqdm(enumerate(data['qa'])):
            total_counts[qa['category']] += 1

            if 'context_match' in qa and len(qa['evidence']) > 0:
                total_context_length = len(qa["context_match"])
                for top_k in [1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200]:
                    recall_counts[top_k][qa['category']] += float(sum([ev in qa["context_match"][:top_k] for ev in qa["evidence"]]))/len(qa['evidence'])

        # hard vs. easy recall
        top_k = [1, 5, 10, 15, 20, 25, 50, 75, 100, 150, 200]
        # top_k_ticks = [float(k_num)/total_context_length for k_num in top_k]
        top_k_ticks = top_k

        # total_hard_counts = sum([total_counts[cat_id] for cat_id in [2,3,4]])
        # total_easy_counts = sum([total_counts[cat_id] for cat_id in [1,5]])

        # print(total_easy_counts, total_hard_counts)
        # print(recall_counts)

        # hard_counts = [float(sum([recall_counts[k][cat_id] for cat_id in [2, 3, 4]]))/total_hard_counts for k in top_k]
        # easy_counts = [float(sum([recall_counts[k][cat_id] for cat_id in [1, 5]]))/total_easy_counts for k in top_k]

        for cat_id in total_counts.keys():
            counts = [float(recall_counts[k][cat_id])/total_counts[cat_id] for k in top_k]
            print(counts)
            plt.plot(top_k_ticks, counts, marker = 'o', linestyle='--', label=labels[cat_id])
        
        # plt.plot(top_k_ticks, hard_counts, marker = 'o', linestyle='--', label="%s; Hard QA" % labels[f_num])
        # plt.plot(top_k_ticks, easy_counts, marker='o', linestyle='--', label="%s; Easy QA" % labels[f_num])


    plt.xlabel('Top-k Retrieved Dialogs')
    plt.ylabel('Retrieval Accuracy')
    plt.title('Retrieval Accuracy by QA Category')
    plt.legend(loc="lower right")
    plt.savefig('./plots_and_txts/category_recall_gpt_3.5_16k.png', dpi=300)


def plot_recall(data_files, labels):

    for f_num, data_file in enumerate(data_files):

        total_counts = defaultdict(lambda: 0)
        hard_memory_counts = defaultdict(lambda: 0)
        easy_memory_counts = defaultdict(lambda: 0)
        recall_counts = {}
        hard_memory_recall = {}
        easy_memory_recall = {}
        for top_k in [1, 5, 10, 15, 20, 25, 50, 75, 100, 150]:
            recall_counts[top_k] = defaultdict(lambda: 0)
            hard_memory_recall[top_k] = defaultdict(lambda: 0)
            easy_memory_recall[top_k] = defaultdict(lambda: 0)

        data = json.load(open(data_file))
        total_context_length = 0
        for i, qa in tqdm(enumerate(data['qa'])):
            total_counts[qa['category']] += 1

            if 'context_match' in qa and len(qa["evidence"]) > 0:
                total_context_length = len(qa["context_match"])
                for top_k in [1, 5, 10, 15, 20, 25, 50, 75, 100, 150]:
                    if all([ev in qa["context_match"][:top_k] for ev in qa["evidence"]]):
                        recall_counts[top_k][qa['category']] += 1
                    for num, ev in enumerate(qa["evidence"]):
                        session_num = int(ev.split(':')[0][1:])
                        if qa['category'] in [1, 5]:
                            if top_k == 1:
                                easy_memory_counts[session_num] += 1
                            if ev in qa["context_match"][:top_k]:
                                easy_memory_recall[top_k][session_num] += 1
                        else:
                            if top_k == 1:
                                hard_memory_counts[session_num] += 1
                            if ev in qa["context_match"][:top_k]:
                                hard_memory_recall[top_k][session_num] += 1

        # hard vs. easy recall
        top_k = [1, 5, 10, 15, 20, 25, 50, 75, 100, 150]
        top_k_ticks = [float(k_num)/total_context_length for k_num in top_k]
        
        total_hard_counts = sum([total_counts[cat_id] for cat_id in [2,3,4]])
        total_easy_counts = sum([total_counts[cat_id] for cat_id in [1,5]])

        print(total_easy_counts, total_hard_counts)
        print(recall_counts)

        hard_counts = [float(sum([recall_counts[k][cat_id] for cat_id in [2, 3, 4]]))/total_hard_counts for k in top_k]
        easy_counts = [float(sum([recall_counts[k][cat_id] for cat_id in [1, 5]]))/total_easy_counts for k in top_k]
        
        plt.plot(top_k_ticks, hard_counts, marker = 'o', linestyle='--', label="%s; Hard QA" % labels[f_num])
        plt.plot(top_k_ticks, easy_counts, marker='o', linestyle='--', label="%s; Easy QA" % labels[f_num])

    plt.xlabel('fraction of recalled context')
    plt.ylabel('Recall Accuracy')
    plt.title('Recall Accuracy for Easy vs. Hard QA')
    plt.legend(loc="upper left")
    plt.savefig('./plots_and_txts/hard_easy_recall.png', dpi=300)

    # plt.clf()
    # # hard
    # min_session = min(list(easy_memory_recall[1].keys()))
    # max_session = max(list(easy_memory_recall[1].keys()))
    # print(easy_memory_recall)
    # print(hard_memory_recall)
    # x = list(range(min_session, max_session + 1))
    # print(x)
    # for k in top_k:
    #     plt.plot(x, [float(hard_memory_recall[k][sess])/hard_memory_counts[sess] for sess in x], marker = 'o', linestyle='--', label="top-k=%s" % k)
    # plt.xlabel('Session Number')
    # plt.ylabel('Recall Accuracy')
    # plt.title('Recall Accuracy for Hard QA at various Top k')
    # plt.legend(loc="upper left")
    # plt.savefig('./plots_and_txts/hard_recall_dpr_by_sess.png', dpi=300)

    # plt.clf()
    # for k in top_k:
    #     plt.plot(x, [float(easy_memory_recall[k][sess])/easy_memory_counts[sess] for sess in x], marker = 'o', linestyle='--', label="top-k=%s" % k)
    # plt.xlabel('Session Number')
    # plt.ylabel('Recall Accuracy')
    # plt.title('Recall Accuracy for Easy QA at various Top k')
    # plt.legend(loc="upper left")
    # plt.savefig('./plots_and_txts/easy_recall_dpr_by_sess.png', dpi=300)


def plot_acc():

    sess = []
    hard_upper_bounds = {'gpt3.5-16k': [0.25, 0.14, 0.67, 0.4, 1.0, 0.34, 0.6, 0.5, 0.4, 0.45, 0.5], 
                         'gpt4-summary': [1.0, 0.14, 0.33, 0.6, 1.0, 0.34, 0.8, 0.7, 1.0, 1.0, 1.0],
                         'gpt3.5-summary': [0.2, 0.1, 0.4, 0.25, 0.5, 0.18, 0.3, 0.4, 0.2, 0.4, 0.5],
                         'llama2-summary': [0.7, 0.14, 0.35, 0.45, 0.7, 0.45, 0.46, 0.56, 0.43, 0.27, 0.65],
                         'gpt4-top-10': [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                         'gpt4-top-25': [0.15, 0.0, 0.1, 0.15, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],
                         'gpt4-top-50': [0.2, 0.14, 0.1, 0.25, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]}
    
    easy_upper_bounds = {'gpt3.5-16k': [0.64, 0.46, 0.65, 0.54, 0.2, 0.62, 0.43, 0.5, 0.5, 0.8, 0.62], 
                         'gpt4-summary': [1.0, 0.73, 0.82, 0.91, 1.0, 0.88, 0.7, 1.0, 0.88, 0.7, 1.0],
                         'gpt3.5-summary': [0.35, 0.44, 0.38, 0.32, 0.18, 0.46, 0.31, 0.25, 0.3, 0.45, 0.24],
                         'llama2-summary': [0.6, 0.7, 0.65, 0.68, 0.7, 0.63, 0.62, 0.5, 0.4, 0.4, 0.5],
                         'gpt4-top-10': [0.2, 0.0, 0.1, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.3, 0.0],
                         'gpt4-top-25': [0.3, 0.16, 0.1, 0.24, 0.2, 0.1, 0.0, 0.0, 0.23, 0.1],
                         'gpt4-top-50': [0.4, 0.45, 0.47, 0.45, 0.48, 0.47, 0.45, 0.45, 0.25, 0.3, 0.2]}

    x = list(range(1, 11 + 1))
    for model in ['gpt3.5-16k', 'gpt4-summary', 'gpt3.5-summary', 'llama2-summary']:
        plt.plot(x, hard_upper_bounds[model], marker = 'o', linestyle='--', label=model)
    top_k = [50]
    for k in top_k:
        plt.plot(x, hard_upper_bounds['gpt4-top-%s' % k], marker = 'o', linestyle='--', label="top-k=%s" % k)
    plt.xlabel('Session Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Hard QA at various Top k')
    plt.legend(loc="upper left")
    plt.savefig('./plots_and_txts/hard_dpr_by_sess.png', dpi=300)

    plt.clf()
    for model in ['gpt3.5-16k', 'gpt4-summary', 'gpt3.5-summary', 'llama2-summary']:
        plt.plot(x, easy_upper_bounds[model], marker = 'o', linestyle='--', label=model)
    top_k = [50]
    for k in top_k:
        plt.plot(x, easy_upper_bounds['gpt4-top-%s' % k], marker = 'o', linestyle='--', label="top-k=%s" % k)
    plt.xlabel('Session Number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for Easy QA at various Top k')
    plt.legend(loc="upper left")
    plt.savefig('./plots_and_txts/easy_dpr_by_sess.png', dpi=300)


def plot_llm_acc():

    sess = []
    hard_upper_bounds = {'gpt3.5-16k': [0, 0.35, 0.16, 0.45, 0.43, 1.0, 0.34, 0.6, 0.5, 0.55, 0.56, 0.60, 0.55], 
                         'gpt4-summary': [0.8, 1.0, 0.14, 0.33, 0.6, 1.0, 0.34, 0.8, 0.7, 1.0, 1.0, 1.0, 1.0],
                         'gpt3.5-summary': [0.3, 0.2, 0.1, 0.4, 0.25, 0.5, 0.18, 0.3, 0.4, 0.2, 0.4, 0.4, 0.3],
                         'llama2-summary': [0.6, 0.7, 0.14, 0.35, 0.45, 0.7, 0.45, 0.46, 0.56, 0.43, 0.27, 0.65, 0.6],
                         'unlimiformer-llama2': [0.1, 0.2, 0.25, 0.35, 0.25, 0.65, 0.55, 0.38, 0.60, 0.45, 0.52, 0.65, 0.65],
                         'streaming-llm-llama2': [0.0, 0.1, 0.12, 0.18, 0.10, 0.25, 0.15, 0.28, 0.35, 0.28, 0.34, 0.45, 0.38]}

    easy_upper_bounds = {'gpt3.5-16k': [0.0, 0.56, 0.46, 0.65, 0.54, 0.2, 0.62, 0.43, 0.5, 0.5, 0.8, 0.6, 0.6], 
                         'gpt4-summary': [1.0, 1.0, 0.73, 0.82, 0.91, 1.0, 0.88, 0.7, 1.0, 0.88, 0.7, 0.8, 0.9],
                         'gpt3.5-summary': [0.38, 0.35, 0.44, 0.38, 0.32, 0.18, 0.46, 0.31, 0.25, 0.3, 0.45, 0.20, 0.34],
                         'llama2-summary': [0.5, 0.6, 0.7, 0.65, 0.68, 0.7, 0.63, 0.62, 0.5, 0.4, 0.4, 0.5, 0.6],
                         'unlimiformer-llama2': [0.1, 0.25, 0.31, 0.29, 0.56, 0.45, 0.51, 0.56, 0.43, 0.27, 0.65, 0.71, 0.65],
                         'streaming-llm-llama2': [0.0, 0.05, 0.14, 0.21, 0.17, 0.13, 0.1, 0.28, 0.35, 0.30, 0.27, 0.32, 0.45]}
    
    for k, v in hard_upper_bounds.items():
        print(len(hard_upper_bounds[k]))
        hard_upper_bounds[k].reverse()
    
    for k, v in easy_upper_bounds.items():
        print(len(easy_upper_bounds[k]))
        easy_upper_bounds[k].reverse()

    x = [n*1500 for n in list(range(1, 13 + 1))]
    # for model in ['gpt4-summary', 'gpt3.5-summary', 'llama2-summary']:
    for model in ['gpt4-summary', 'gpt3.5-16k', 'unlimiformer-llama2', 'streaming-llm-llama2']:
        plt.plot(x, hard_upper_bounds[model], marker = 'o', linestyle='--', label=model, linewidth=1)
    plt.xlabel('Content Window Length')
    plt.ylabel('Accuracy')
    plt.title('Accuracy with Increasing Context Length')
    plt.legend(loc="upper right")
    plt.savefig('./plots_and_txts/hard_context_window.png', dpi=300)

    plt.clf()
    # for model in ['gpt4-summary', 'gpt3.5-summary', 'llama2-summary']:
    for model in ['gpt4-summary', 'gpt3.5-16k', 'unlimiformer-llama2', 'streaming-llm-llama2']:
        plt.plot(x, easy_upper_bounds[model], marker = 'o', linestyle='--', label=model, linewidth=1)
    plt.xlabel('Content Window Length')
    plt.ylabel('Accuracy')
    plt.title('Accuracy with Increasing Context Length')
    plt.legend(loc="upper right")
    plt.savefig('./plots_and_txts/easy_context_window.png', dpi=300)

def plot_retrieval_acc():

    top_k = [1, 5, 10, 15, 20, 25, 50, 75, 100, 150]
    hard_acc = [0, 0, 0, 0, 0, 0.07, 0.09, 0.12, 0.15, 0.32]
    easy_acc = [0, 0, 0.04, 0.06, 0.08, 0.08, 0.15, 0.25, 0.56, 0.61]

    x = [n*1500 for n in list(range(1, 13 + 1))]
    # for model in ['gpt4-summary', 'gpt3.5-summary', 'llama2-summary']:
    plt.plot(top_k, easy_acc, marker = 'o', linestyle='--', label='easy', linewidth=1)
    plt.plot(top_k, hard_acc, marker = 'o', linestyle='--', label='hard', linewidth=1)
    plt.xlabel('Top-K Retrieved Context')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.title('Accuracy with Increasing Retrieved Context')
    plt.legend(loc="upper right")
    plt.savefig('./plots_and_txts/retrieval_acc.png', dpi=300)

def run_eval(data, out_file, model, top_k_recall=5):

    for i, qa in tqdm(enumerate(data['qa'])):

        if model not in qa:

            if model == 'gpt4':
                time.sleep(5)

            if qa['category'] == 3:
                question = 'Today is 30 May, 2020.' + qa['question'] + ' Give an approximate date, month or year.'
            else:
                question = qa['question']

            query = conv + '\n' + 'Based on the above conversations, answer the following question.\nQuestion: ' + question + '\n' + 'Answer: '
            answer = run_chatgpt(query, num_gen=1, num_tokens_request=50, 
                    model=model, use_16k=use_16k, temperature=0, wait_time=10)
            
            data['qa'][i]['gpt4'] = answer.strip()

            with open(out_file, 'w') as f:
                json.dump(data, f, indent=2)


def get_chatgpt_answers(args, ann_file, out_file, model='chatgpt', use_16k=False, overwrite=False, captions=None):

    if args.reader == 'chatgpt':
        set_openai_key(args)

    QA_PROMPT = """
    {}

    Answer the following question in a short phrase. If no information is available to answer the question, write 'No information available':

    Question: {} Short answer:
    """

    from generative_agents.conversation_utils import run_chatgpt
    data = json.load(open(ann_file))

    if os.path.exists(out_file):
        data = json.load(open(out_file))

    conv = {}
    for i in range(1,20):
        if 'session_%s' % i in data:
            date_time_string = data['session_%s_date_time' % i]
            for dialog in data['session_%s' % i]:
                    try:
                        caption = captions['session_%s/a/%s' % (i, dialog["img_file"][0])]
                        conv[dialog["dia_id"]] = '(' + date_time_string + ') ' + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\" and shared ' + caption
                    except:
                        conv[dialog["dia_id"]] = '(' + date_time_string + ') ' + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'

    
    for i, qa in tqdm(enumerate(data['qa'])):

        for top_k in [10, 25, 50, 75]:

            if 'prediction_%s' % top_k not in qa or overwrite:

                if model == 'gpt4':
                    time.sleep(5)
                elif model == 'chatgpt':
                    time.sleep(0.5)

                if qa['category'] == 3:
                    question = qa['question'] + 'Use DATE of CONVERSATION to give an approximate date, month or year.'
                else:
                    question = qa['question']

                retrieved_context = [conv[dia_id] for dia_id in qa["context_match"][:top_k]]
                retrieved_context = '\n'.join(retrieved_context)

                query = QA_PROMPT.format(retrieved_context, question)
                answer = run_chatgpt(query, num_gen=1, num_tokens_request=50, 
                        model=model, use_16k=use_16k, temperature=0, wait_time=10)
                
                data['qa'][i]['prediction_%s' % top_k] = answer.strip()

                with open(out_file, 'w') as f:
                    json.dump(data, f, indent=2)


def main():

    args = parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    context_tokenizer, context_model, question_tokenizer, question_model = init_model(args)

    # ann_file = './data/multimodal_dialog/data/3.json'
    # data = json.load(open(ann_file))
    # captions = json.load(open('./data/multimodal_dialog/generated/3/captions.json'))
    # context_ids, context_embeddings = get_context_embeddings(args, data, context_tokenizer, context_model, captions)

    # out_file = os.path.join(args.out_dir, '3.json')
    # run_recall_eval(args, data, out_file, context_embeddings, question_model, question_tokenizer, context_ids)
# # 

    # plot_recall(['./outputs/dpr/3.json', './outputs/dragon/3.json', './outputs/contriever/3.json'], ['DPR', 'Dragon', 'Contriever'])

    data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json')]
    # data_files = ['./data/multimodal_dialog/data/3.json']

    ems = 0
    for f in data_files:

        break

        data = json.load(open(f))
        context_ids, context_embeddings = get_context_embeddings(args, data, context_tokenizer, context_model)
        out_file = os.path.join(args.out_dir, f.split('/')[-1])
        run_recall_eval(args, data, out_file, context_embeddings, question_model, question_tokenizer, context_ids)

        break

        # get_chatgpt_answers(args, f, os.path.join(args.out_dir, os.path.split(f)[-1]), model=args.reader, use_16k=False, overwrite=False)
        # for top_k in [10, 25, 50, 75]:
        #     exact_matches, lengths = eval_question_answering(os.path.join(args.out_dir, os.path.split(f)[-1]), 'prediction_%s' % top_k)
        #     em = round(sum(exact_matches)/len(exact_matches), 4)
        #     ems += em
        #     save_eval(os.path.join(args.out_dir, os.path.split(f)[-1]), exact_matches, 'exact_match_%s' % top_k)
        
        for top_k in [10, 25, 50, 75]:
            analyze_acc(os.path.join(args.out_dir, os.path.split(f)[-1]).replace('.json', '_scores.json'), 'exact_match_%s' % top_k)
    
    print("Exact Match Acc.: ", ems)

    # analyze_recall(out_file)
    # plot_acc()
    # plot_llm_acc()
    # plot_retrieval_acc()
    plot_recall_by_category(['./outputs/dragon/26_post_qa.json'], {1: 'Multi-Hop', 2: 'Time', 3: 'Knowledge', 4: 'Single-Hop', 5: 'Adversarial'})

if __name__ == "__main__":
    main()