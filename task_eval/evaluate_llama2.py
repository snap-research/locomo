import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import random
import os, json
from tqdm import tqdm
import time
import argparse
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_acc
from transformers import AutoTokenizer
import transformers
import torch
import huggingface_hub

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
import huggingface_hub


MAX_LENGTH={'llama2': 4096,
            'llama2-70b': 4096,
            'llama2-chat': 4096,
            'llama2-chat-70b': 4096,
            'llama3-chat-70b': 4096,
            'gpt-3.5-turbo-16k': 16000,
            'gpt-3.5-turbo': 4096,
            'gemma-7b-it': 8000,
            'mistral-7b-128k': 128000,
            'mistral-7b-4k': 4096,
            'mistral-7b-8k': 8000,
            'mistral-instruct-7b-4k': 4096,
            'mistral-instruct-7b-8k': 8000,
            'mistral-instruct-7b-32k-v2': 8000,
            'mistral-instruct-7b-8k-new': 8000,
            'mistral-instruct-7b-32k': 32000,
            'mistral-instruct-7b-128k': 128000}


QA_PROMPT = """
Based on the above conversations, write a short answer for the following question in a few words. Do not write complete and lengthy sentences. Answer with exact words from the conversations whenever possible.

Question: {}
"""

# QA_PROMPT_BATCH = """
# Based on the above conversations, answer the following questions in a few words. Write the answers as a list of strings in the json format. Start and end with a square bracket.

# """

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the question number as 'key' and the short answer as value. Answer with exact words from the conversations whenever possible.

"""

LLAMA2_CHAT_SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation.
If you don't know the answer to a question, please don't share false information.
<</SYS>>

{} [/INST]
"""


LLAMA3_CHAT_SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation.
If you don't know the answer to a question, please don't share false information.
<</SYS>>

{} [/INST]
"""


MISTRAL_INSTRUCT_SYSTEM_PROMPT = """
<s>[INST] {} [/INST]
"""

GEMMA_INSTRUCT_PROMPT = """
<bos><start_of_turn>user
{}<end_of_turn>
"""

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"

ANS_TOKENS_PER_QUES = 50


def run_mistral(pipeline, question, data, tokenizer, args):

    question_prompt =  QA_PROMPT.format(question)
    query_conv = get_input_context(data, MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(question_prompt), tokenizer, args)

    # without chat_template
    # query = MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)
    # with chat template
    query = tokenizer.apply_chat_template([{"role": "user", "content": query_conv + '\n\n' + question_prompt}], tokenize=False, add_generation_prompt=True)

    sequences = pipeline(
                        query,
                        # max_length=8000,
                        max_new_tokens=args.batch_size*ANS_TOKENS_PER_QUES,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=10,
                        temperature=0.4,
                        top_p=0.9,
                        return_full_text=False,
                        num_return_sequences=1,
                        )
    return sequences[0]['generated_text']


def run_gemma(pipeline, question, data, tokenizer, args):

    question_prompt =  QA_PROMPT.format(question)
    query_conv = get_input_context(data, GEMMA_INSTRUCT_PROMPT.format(question_prompt), tokenizer, args)

    # without chat_template
    # query = MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)
    # with chat template
    query = tokenizer.apply_chat_template([{"role": "user", "content": query_conv + '\n\n' + question_prompt}], tokenize=False, add_generation_prompt=True)

    sequences = pipeline(
                        query,
                        # max_length=8000,
                        max_new_tokens=args.batch_size*ANS_TOKENS_PER_QUES,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=10,
                        temperature=0.4,
                        top_p=0.9,
                        return_full_text=False,
                        num_return_sequences=1,
                        )
    return sequences[0]['generated_text']


def run_llama(pipeline, question, data, tokenizer, args):

    question_prompt =  QA_PROMPT.format(question)
    query_conv = get_input_context(data, LLAMA3_CHAT_SYSTEM_PROMPT.format(question_prompt), tokenizer, args)

    # without chat_template
    # query = MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)
    # with chat template
    query = tokenizer.apply_chat_template([{"role": "system", "content": "You are a helpful, respectful and honest assistant whose job is to understand the following conversation and answer questions based on the conversation. If you don't know the answer to a question, please don't share false information."},
                                           {"role": "user", "content": query_conv + '\n\n' + question_prompt}], tokenize=False, add_generation_prompt=True)

    sequences = pipeline(
                        query,
                        # max_length=8000,
                        max_new_tokens=args.batch_size*ANS_TOKENS_PER_QUES,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=10,
                        temperature=0.4,
                        top_p=0.9,
                        return_full_text=False,
                        num_return_sequences=1,
                        )
    return sequences[0]['generated_text']


def save_eval(data_file, accs, key='exact_match'):

    data = json.load(open(data_file))
    assert len(data['qa']) == len(accs), (len(data['qa']), len(accs), accs)
    if os.path.exists(data_file.replace('.json', '_scores.json')):
        data = json.load(open(data_file.replace('.json', '_scores.json')))
    for i in range(0, len(data['qa'])):
        data['qa'][i][key] = accs[i]
    with open(data_file.replace('.json', '_scores.json'), 'w') as f:
        json.dump(data, f, indent=2)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--openai-key-file', type=str, default='')
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--use-4bit', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args()
    return args


def get_chatgpt_summaries(ann_file):

    data = json.load(open(ann_file))
    conv = ''
    for i in range(1,20):
        if 'session_%s' % i in data:
            conv = conv + data['session_%s_date_time' % i] + '\n'
            for dialog in data['session_%s' % i]:
                conv = conv + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'


def get_input_context(data, question_prompt, encoding, args):

    # get number of tokens from question prompt
    question_tokens = len(encoding.encode(question_prompt))

    # start instruction prompt
    speakers_names = list(set([d['speaker'] for d in data['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = len(encoding.encode(start_prompt))

    query_conv = ''
    total_tokens = 0
    min_session = -1
    stop = False
    max_session = [i for i in range(1, 50) if 'session_%s' % i in data and data['session_%s' % i] != []][-1]
    for i in range(max_session, 0, -1):
        if 'session_%s' % i in data:
            for dialog in data['session_%s' % i][::-1]:
                turn = ''
                try:
                    turn = dialog['speaker'] + ' said, \"' + dialog['compressed_text'] + '\"'
                except KeyError:
                    turn = dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"'
                if "img_file" in dialog and len(dialog["img_file"]) > 0:
                    turn += ' [shares %s]' % dialog["blip_caption"]
                turn = turn + '\n'

                # get an approximate estimate of where to truncate conversation to fit into contex window
                new_tokens = len(encoding.encode('DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + turn))
                if (start_tokens + new_tokens + total_tokens + question_tokens) < (MAX_LENGTH[args.model]-(ANS_TOKENS_PER_QUES*args.batch_size)): # if new turns still fit into context window, add to query
                    query_conv = turn + query_conv
                    total_tokens += len(encoding.encode(turn))
                else:
                    min_session = i
                    stop = True
                    break

            query_conv = '\nDATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv
        
        if stop:
            break
    
    query_conv = start_prompt + query_conv
    
    if min_session == -1:
        print("Saved %s tokens in query conversation from full conversation" % total_tokens)
    else:
        print("Saved %s tokens in query conversation from %s out of %s sessions" % (total_tokens, max_session-min_session, max_session))

    return query_conv


def get_answers(ann_file, out_file, args, pipeline, model_name):

    if 'mistral' in model_name:
        encoding = AutoTokenizer.from_pretrained(model_name)
    else:
        encoding = AutoTokenizer.from_pretrained(model_name)
    data = json.load(open(ann_file))

    if os.path.exists(out_file):
        data = json.load(open(out_file))

    for batch_start_idx in range(0, len(data['qa']) + args.batch_size, args.batch_size):

        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            # end if all questions have been included
            if i>=len(data['qa']):
                break
            qa = data['qa'][i]
            # skip if already predicted and overwrite is set to False            
            if '%s_prediction' % args.model not in qa or args.overwrite:
                include_idxs.append(i)
            else:
                print("Skipping -->", qa['question'])
                continue

            # pre-processing steps for Temporal (2) and Adversarial (5) categories
            if qa['category'] == 2:
                questions.append(qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.')
            elif qa['category'] == 5:
                question = qa['question'] + " (a) {} (b) {}. Select the correct answer by writing (a) or (b)."
                if random.random() < 0.5:
                    question = question.format('No information available', qa['answer'])
                    answer = {'a': 'No information available', 'b': qa['answer']}
                else:
                    question = question.format(qa['answer'], 'No information available')
                    answer = {'b': 'No information available', 'a': qa['answer']}
                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
                # questions.append(qa['question'] + " Write NOT ANSWERABLE if the question cannot be answered.")
            else:
                questions.append(qa['question'])

        if questions == []:
            continue


        if args.batch_size == 1:

            # ######################################################################################################################
            # question_prompt =  QA_PROMPT.format(questions[0])
            # if 'chat' in args.model:
            #     query_conv = get_input_context(data, LLAMA2_CHAT_SYSTEM_PROMPT.format(question_prompt), encoding, ann_file, args)
            # elif 'instruct' in args.model:
            #     query_conv = get_input_context(data, MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(question_prompt), encoding, ann_file, args)
            # else:
            #     query_conv = get_input_context(data, question_prompt, encoding, ann_file, args)
            
            # if 'chat' in args.model:
            #     query = LLAMA2_CHAT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)
            # elif 'instruct' in args.model:
            #     # without chat-style input for conversations
            #     # query = MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)
            #     # with chat-style input for conversations
            #     query = MISTRAL_INSTRUCT_SYSTEM_PROMPT.format(query_conv + '\n\n' + question_prompt)

            # else:
            #     query = query_conv + '\n' + question_prompt
            
            # print(query)
            # if 'mistral' not in model_name:
            #     sequences = pipeline(
            #                     query,
            #                     do_sample=True,
            #                     top_k=10,
            #                     num_return_sequences=1,
            #                     eos_token_id=encoding.eos_token_id,
            #                     max_new_tokens=args.batch_size*batch_multiplier,
            #                     return_full_text=False
            #                 )
            # else:
            #     sequences = pipeline(
            #                     query,
            #                     # max_length=8000,
            #                     max_new_tokens=args.batch_size*batch_multiplier,
            #                     pad_token_id=encoding.pad_token_id,
            #                     eos_token_id=encoding.eos_token_id,
            #                     do_sample=True,
            #                     top_k=10,
            #                     temperature=0.4,
            #                     top_p=0.9,
            #                     return_full_text=False,
            #                     num_return_sequences=1,
            #                 )
            
            # print(sequences[0]['generated_text'])
            # answer = sequences[0]['generated_text'].replace('\\"', "'").strip()
            # ####################################################################################################

            if 'mistral' in model_name:
                answer = run_mistral(pipeline, questions[0], data, encoding, args)
            elif 'llama' in model_name:
                answer = run_llama(pipeline, questions[0], data, encoding, args)
            elif 'gemma' in model_name:
                answer = run_gemma(pipeline, questions[0], data, encoding, args)
            else:
                raise NotImplementedError
            
            print(questions[0], answer)

            # post process answers, necessary for Adversarial Questions
            answer = answer.replace('\\"', "'").strip()
            answer = [w.strip() for w in answer.split('\n') if not w.strip().isspace()][0]
            if len(cat_5_idxs) > 0:
                answer = answer.lower().strip()
                if '(a)' in answer:
                    answer = cat_5_answers[0]['a']
                else:
                    answer = cat_5_answers[0]['b']
            else:
                answer = answer.lower().replace('(a)', '').replace('(b)', '').replace('a)', '').replace('b)', '').replace('answer:', '').strip()
            data['qa'][batch_start_idx]['%s_prediction' % args.model] = answer

        else:            
            raise NotImplementedError

        # save after every batch for backup in case process terminates
        with open(out_file, 'w') as f:
            json.dump(data, f, indent=2)


def main():

    # get arguments
    args = parse_args()

    # output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json')]
    # data_files = [os.path.join(args.data_dir, f) for f in ['44_post_qa_post_clean_adv.json']]
    data_files = [os.path.join(args.data_dir, f) for f in ['41_post_qa_post_clean_adv.json']]

    # data_files = [os.path.join(args.data_dir, f) for f in ['42_post_qa_post_clean_adv.json',
    #                                                        '43_post_qa_post_clean_adv_new.json', 
    #                                                        '44_post_qa_post_clean_adv_new.json',
    #                                                        '47_post_qa_post_clean_adv_new.json',
    #                                                        '48_post_qa_post_clean_adv_new.json',
    #                                                        '49_post_qa_post_clean_adv_new.json',
    #                                                        '50_post_qa_post_clean_adv_new.json']]

    if args.model == 'llama2':
        model_name = "meta-llama/Llama-2-7b-hf"

    elif args.model == 'llama2-70b':
        model_name = "meta-llama/Llama-2-70b-hf"

    elif args.model == 'llama2-chat':
        model_name = "meta-llama/Llama-2-7b-chat-hf"

    elif args.model == 'llama2-chat-70b':
        model_name = "meta-llama/Llama-2-70b-chat-hf"

    elif args.model == 'llama3-chat-70b':
        model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

    elif args.model in ['mistral-7b-128k', 'mistral-7b-4k', 'mistral-7b-8k']:
        model_name = "mistralai/Mistral-7B-v0.1"

    elif args.model in ['mistral-instruct-7b-128k', 'mistral-instruct-7b-8k', 'mistral-instruct-7b-12k']:
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"

    elif args.model in ['mistral-instruct-7b-8k-new']:
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    elif args.model in ['mistral-instruct-7b-32k-v2']:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    elif args.model in ['gemma-7b-it']:
        model_name = 'google/gemma-7b-it'

    else:
        raise ValueError
    
    # hf_token = "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"
    # hf_token = "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"
    hf_token = "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"
    huggingface_hub.login(hf_token)

    if args.use_4bit:

        print("Using 4-bit inference")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token="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")
        tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

        if 'gemma' in args.model:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                            bnb_4bit_compute_dtype=torch.float16)
        else:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        if 'mistralai' in model_name:
            if 'v0.1' in model_name:
                model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                            torch_dtype=torch.float16, 
                                                            attn_implementation="flash_attention_2",
                                                            quantization_config=bnb_config,
                                                            device_map="auto",
                                                            trust_remote_code=True,)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name,
                                                            quantization_config=bnb_config,
                                                            device_map="auto",
                                                            trust_remote_code=True)
        
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                                            torch_dtype=torch.float16,
                                            quantization_config=bnb_config,
                                            device_map="auto",
                                            trust_remote_code=True,)

        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
            device_map="auto",    # finds GPU
        )
    
    else:
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # pipeline = None
    
    print("Loaded model")

    ems = []
    for f in data_files:
        get_answers(f, os.path.join(args.out_dir, os.path.split(f)[-1]), args, pipeline, model_name)
        exact_matches, lengths, recalls = eval_question_answering(os.path.join(args.out_dir, os.path.split(f)[-1]), '%s_prediction' % args.model)
        ems.extend(exact_matches)
        save_eval(os.path.join(args.out_dir, os.path.split(f)[-1]), exact_matches, args.model + '_rouge')
        analyze_acc(os.path.join(args.out_dir, os.path.split(f)[-1]).replace('.json', '_scores.json'), 
                    os.path.join(args.out_dir, os.path.split(f)[-1]).replace('.json', '_score_stats.json'),
                    args.model,
                    args.model + '_rouge')
    
    print("Exact Match Acc.: ", sum(ems)/len(ems))
    
    # get_chatgpt_answers('./data/multimodal_dialog/completed_annotations/3.json', 
    #                     './data/multimodal_dialog/completed_annotations/3_out_gpt4_summary.json', 
    #                     summary=True, model='gpt4')


main()

