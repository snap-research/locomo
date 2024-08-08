import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import random
import os, json
from tqdm import tqdm
import time
import argparse
from global_methods import run_claude, set_anthropic_key
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_acc

from scipy.spatial import distance
import numpy as np

MAX_LENGTH={'claude-sonnet': 2000000, 'claude-haiku': 2000000}
PER_QA_TOKEN_BUDGET = 50

QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""

# QA_PROMPT_BATCH = """
# Based on the above conversations, answer the following questions in a few words. Write the answers as a list of strings in the json format. Start and end with a square bracket.

# """

QA_PROMPT_BATCH = """
Based on the above conversations, write short answers for each of the following questions in a few words. Write the answers in the form of a json dictionary where each entry contains the string format of question number as 'key' and the short answer as value. Use single-quote characters for named entities. Answer with exact words from the conversations whenever possible.

"""

# If no information is available to answer the question, write 'No information available'.

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def process_ouput(text):

    # single_quote_count = text.count("'")
    # double_quote_count = text.count('"')
    # if single_quote_count > double_quote_count:
    #     text = text.replace('"', "")
    #     text = text.replace("'", '"')
    #     return json.loads(text)
    # else:
    #     return json.loads(text)

    text = text.strip()
    if text[0] != '{':
        start = text.index('{')
        text = text[start:].strip()

    return json.loads(text)


def get_cat_5_answer(model_prediction, answer_key):

    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if 'a' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    elif len(model_prediction) == 3:
        if '(a)' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    else:
        return model_prediction


def save_eval(data_file, accs, key, recall_accs=None):

    data = json.load(open(data_file))
    assert len(data['qa']) == len(accs), (len(data['qa']), len(accs), accs)
    if len(recall_accs) > 0:
        assert len(data['qa']) == len(recall_accs)
    if os.path.exists(data_file.replace('.json', '_scores.json')):
        data = json.load(open(data_file.replace('.json', '_scores.json')))
    for i in range(0, len(data['qa'])):
        data['qa'][i][key] = accs[i]
        if len(recall_accs) > 0:
            data['qa'][i][key + '_recall'] = recall_accs[i]
    with open(data_file.replace('.json', '_scores.json'), 'w') as f:
        json.dump(data, f, indent=2)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--use-rag', action="store_true")
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--rag-mode', type=str, default="")
    parser.add_argument('--prompt-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args()
    return args

def get_input_context(data, num_question_tokens, model, args):

    query_conv = ''
    min_session = -1
    stop = False
    max_session = [i for i in range(1, 50) if 'session_%s' % i in data and data['session_%s' % i] != []][-1]
    for i in range(max_session, 0, -1):
        if 'session_%s' % i in data:
            query_conv += "\n\n"
            for dialog in data['session_%s' % i][::-1]:
                turn = ''
                # conv = conv + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                try:
                    turn = dialog['speaker'] + ' said, \"' + dialog['compressed_text'] + '\"' 
                    # conv = conv + dialog['speaker'] + ': ' + dialog['compressed_text'] + '\n'
                except KeyError:
                    turn = dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' 
                    # conv = conv + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'
                if "img_file" in dialog and len(dialog["img_file"]) > 0 and dialog["blip_caption"] != "":
                    turn += ' [shares %s]' % dialog["blip_caption"]
                turn += '\n'
        
                # num_tokens = len(model.count_tokens('DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + turn))
                # if (num_tokens + len(model.count_tokens(query_conv)) + num_question_tokens) < (MAX_LENGTH[args.model]-(PER_QA_TOKEN_BUDGET*(args.batch_size))): # 20 tokens assigned for answers
                #     query_conv = turn + query_conv
                # else:
                #     min_session = i
                #     stop = True
                #     break

                query_conv = turn + query_conv

            query_conv = '\nDATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv
        if stop:
            break
        
        # if min_session == -1:
        #     print("Saved %s tokens in query conversation from full conversation" % len(model.count_tokens(query_conv)))
        # else:
        #     print("Saved %s conv. tokens + %s question tokens in query from %s out of %s sessions" % (len(model.count_tokens(query_conv)), num_question_tokens, max_session-min_session, max_session))

    return query_conv


def get_answers(ann_file, out_file, args):


    in_data = json.load(open(ann_file))


    if os.path.exists(out_file):
        out_data = json.load(open(out_file))
    else:
        out_data = in_data.copy()

    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))

    # start instruction prompt
    speakers_names = list(set([d['speaker'] for d in in_data['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    # start_tokens = model.count_tokens(start_prompt)
    start_tokens =100

    if args.rag_mode:
        raise NotImplementedError
    else:
        context_database, query_vectors = None, None

    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)

    for batch_start_idx in range(0, len(in_data['qa']), args.batch_size):

        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            if i>=len(in_data['qa']):
                break

            qa = in_data['qa'][i]
            
            if prediction_key not in out_data['qa'][i] or args.overwrite:
                include_idxs.append(i)
            else:
                continue

            if qa['category'] == 2:
                questions.append(qa['question'] + ' Use DATE of CONVERSATION to answer with an approximate date.')
            elif qa['category'] == 5:
                question = qa['question'] + " Select the correct answer: (a) {} (b) {}. "
                if random.random() < 0.5:
                    question = question.format('Not mentioned in the conversation', qa['answer'])
                    answer = {'a': 'Not mentioned in the conversation', 'b': qa['answer']}
                else:
                    question = question.format(qa['answer'], 'Not mentioned in the conversation')
                    answer = {'b': 'Not mentioned in the conversation', 'a': qa['answer']}

                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
                # questions.append(qa['question'] + "Write NOT ANSWERABLE if the question cannot be answered")
            else:
                questions.append(qa['question'])


        if questions == []:
            continue


        context_ids = None
        if args.use_rag:
            
            raise NotImplementedError
        else:
            question_prompt =  QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            # num_question_tokens = model.count_tokens(question_prompt)
            num_question_tokens=100
            query_conv = get_input_context(in_data, num_question_tokens + start_tokens, None, args)
            query_conv = start_prompt + query_conv
        

        # print("%s tokens in query" % len(model.count_tokens(query_conv)))

        if 'pro-1.0' in args.model:
            time.sleep(5)

        if args.batch_size == 1:

            query = query_conv + '\n\n' + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + QA_PROMPT_CAT_5.format(questions[0])
            print('------------------------------------------------------------------------------------')
            print(query)
            print('------------------------------------------------------------------------------------')
            answer = run_claude(query, PER_QA_TOKEN_BUDGET, args.model)
            
            if len(cat_5_idxs) > 0:
                answer = get_cat_5_answer(answer, cat_5_answers[0])
            print('------------------------------------------------------------------------------------')
            print(answer)
            print('------------------------------------------------------------------------------------')

            out_data['qa'][include_idxs[0]][prediction_key] = answer.strip()
            if args.use_rag:
                out_data['qa'][include_idxs[0]][prediction_key + '_context'] = context_ids

        else:
            # query = query_conv + '\n' + QA_PROMPT_BATCH + "\n".join(["QUESTION: %s" % q for q in questions])
            query = query_conv + '\n' + question_prompt
            # print(query)
            
            trials = 0
            while trials < 5:
                try:
                    trials += 1
                    print("Trial %s" % trials)
                    # print("Sending query of %s tokens" % len(model.count_tokens(query)))
                    print("Trying with answer token budget = %s per question" % PER_QA_TOKEN_BUDGET)
                    answer = run_claude(query, PER_QA_TOKEN_BUDGET * args.batch_size, args.model)
                    answer = answer.replace('\\"', "'").replace('json','').replace('`','').strip()
                    print(answer)
                    # try:
                    #     answers = json.loads(answer.strip())
                    # except:
                    answers = process_ouput(answer.strip())
                    break
                except json.decoder.JSONDecodeError:
                    pass
            
            for k, idx in enumerate(include_idxs):
                try:
                    answers = process_ouput(answer.strip())
                    # answers = json.loads(answer.strip())
                    # data['qa'][idx]['%s_prediction' % args.model] = answers[k]['answer'].strip()
                    if k in cat_5_idxs:
                        predicted_answer = get_cat_5_answer(answers[str(k)], cat_5_answers[cat_5_idxs.index(k)])
                        out_data['qa'][idx][prediction_key] = predicted_answer
                    else:
                        try:
                            out_data['qa'][idx][prediction_key] = str(answers[str(k)]).replace('(a)', '').replace('(b)', '').strip()
                        except:
                            out_data['qa'][idx][prediction_key] = ', '.join([str(n) for n in list(answers[str(k)].values())])
                except:
                    try:
                        answers = json.loads(answer.strip())
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answers[k], cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = answers[k].replace('(a)', '').replace('(b)', '').strip()
                    except:
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answer.strip(), cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = json.loads(answer.strip().replace('(a)', '').replace('(b)', '').split('\n')[k])[0]

        with open(out_file, 'w') as f:
            json.dump(out_data, f, indent=2)


def main():

    # get arguments
    args = parse_args()

    print("******************  Evaluating Model %s ***************" % args.model)

    # set openai API key
    set_anthropic_key()

    # output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json') if '26' in f] # fix for other files
    # data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if 'Chat_8' in f] # fix for other files
    data_files = [os.path.join(args.data_dir, f) for f in ['48_post_qa_post_clean_adv.json']]

    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)

    ems = []
    total = 0
    for f in data_files:
        get_answers(f, os.path.join(args.out_dir, os.path.split(f)[-1]), args)
        exact_matches, lengths, recall = eval_question_answering(os.path.join(args.out_dir, os.path.split(f)[-1]), prediction_key)
        ems.extend(exact_matches)
        save_eval(os.path.join(args.out_dir, os.path.split(f)[-1]), exact_matches, "%s_f1" % args.model if not args.use_rag else "%s_%s_top_%s_f1" % (args.model, args.rag_mode, args.top_k), recall)
        analyze_acc(os.path.join(args.out_dir, os.path.split(f)[-1]).replace('.json', '_scores.json'), 
                    os.path.join(args.out_dir, os.path.split(f)[-1]).replace('.json', '_score_stats.json'),
                    args.model if not args.use_rag else "%s_%s_top_%s" % (args.model, args.rag_mode, args.top_k),
                    "%s_f1" % args.model if not args.use_rag else "%s_%s_top_%s_f1" % (args.model, args.rag_mode, args.top_k), rag=args.use_rag)
        # encoder=tiktoken.encoding_for_model(args.model))
    
    print("Exact Match Acc.: ", sum(ems)/len(ems))
    
    # get_chatgpt_answers('./data/multimodal_dialog/completed_annotations/3.json', 
    #                     './data/multimodal_dialog/completed_annotations/3_out_gpt4_summary.json', 
    #                     summary=True, model='gpt4')


main()

