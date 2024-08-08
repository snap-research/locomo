import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import random
import os, json
from tqdm import tqdm
import time
import argparse
from global_methods import run_chatgpt, set_openai_key
from task_eval.dpr_qa import get_embeddings
from task_eval.evaluation import eval_question_answering
from task_eval.evaluation_stats import analyze_acc
from generative_agents.memory_utils import get_session_facts
import tiktoken
import openai

from scipy.spatial import distance
import numpy as np

MAX_LENGTH={'gpt-4-turbo': 128000,
            'gpt-4': 4096,
            'gpt-3.5-turbo-16k': 16000,
            'gpt-3.5-turbo-12k': 12000,
            'gpt-3.5-turbo-8k': 8000,
            'gpt-3.5-turbo-4k': 4000,
            'gpt-3.5-turbo': 4096,
            'gpt-4-32k': 320000}
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
Based on the above conversations, write short answers for each of the following questions in a few words. 
Write the answers in the form of a json dictionary where each entry contains the question number as "key" and the short answer as "value". 
Use single-quote characters for named entities and double-quote characters for enclosing json elements. Answer with exact words from the conversations whenever possible.

"""

# If no information is available to answer the question, write 'No information available'.

CONV_START_PROMPT = "Below is a conversation between two people: {} and {}. The conversation takes place over multiple days and the date of each conversation is wriiten at the beginning of the conversation.\n\n"


def process_ouput(text):

    single_quote_count = text.count("'")
    double_quote_count = text.count('"')
    if single_quote_count > double_quote_count:
        text = text.replace('"', "")
        text = text.replace("'", '"')
        print(text)
        return json.loads(text)
    else:
        return json.loads(text)


def prepare_for_rag(args, data, ann_file):

    if args.use_rag and args.rag_mode == "summary":

        # check if embeddings exist
        if not os.path.exists(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_summary.pkl'))):

            summaries = []
            date_times = []
            context_ids = []
            for i in range(1,50):

                if 'session_%s' % i not in data:
                    break
                if len(data['session_%s' % i]) == 0:
                    continue

                date_time = data['session_%s_date_time' % i]
                if 'session_%s_summary' % i not in data:
                    conv = ''
                    conv = conv + data['session_%s_date_time' % i] + '\n'
                    for dialog in data['session_%s' % i]:
                        conv = conv + dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                    query = "Generate a concise summary of the following conversation using exact words from the conversation wherever possible. The summary should contain all facts about the two speakers, as well as references to time.\n"
                    query = query + conv + "\n"
                    print("Getting summary for session %s" % i)
                    session_summary = run_chatgpt(query, num_gen=1, num_tokens_request=256, 
                                                model='chatgpt', use_16k=False, 
                                                temperature=1.0, wait_time=2)
                    data['session_%s_summary' % i] = session_summary
                    with open(ann_file, 'w') as f:
                        json.dump(data, f, indent=2)
                else:
                    session_summary = data['session_%s_summary' % i]
                summaries.append(session_summary)
                date_times.append(date_time)
                context_ids.append('S%s'%i)

            print("Getting embeddings for %s summaries" % len(summaries))
            embeddings = get_embeddings(args, summaries, 'context')
            print(embeddings.shape)
            database = {'embeddings': embeddings,
                                'date_time': date_times,
                                'dia_id': context_ids,
                                'context': summaries}

            with open(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_summary.pkl')), 'wb') as f:
                pickle.dump(database, f)
        else:
            database = pickle.load(open(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_summary.pkl')), 'rb'))


    if args.use_rag and args.rag_mode == 'dialog':
        # check if embeddings exist
        if not os.path.exists(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_dialog.pkl'))):

            dialogs = []
            date_times = []
            context_ids = []
            for i in range(1,50):

                if 'session_%s' % i not in data:
                    break
                if len(data['session_%s' % i]) == 0:
                   continue
            
                date_time = data['session_%s_date_time' % i]
                for dialog in data['session_%s' % i]:
                    context_ids.append(dialog['dia_id'])
                    date_times.append(date_time)
                    dialogs.append(dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"')

            print("Getting embeddings for %s dialogs" % len(dialogs))
            embeddings = get_embeddings(args, dialogs, 'context')
            print(embeddings.shape)
            database = {'embeddings': embeddings,
                             'date_time': date_times,
                             'dia_id': context_ids,
                             'context': dialogs}

            with open(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_dialog.pkl')), 'wb') as f:
                pickle.dump(database, f)

        else:
            database = pickle.load(open(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_dialog.pkl')), 'rb'))

    if args.use_rag and args.rag_mode == 'observation':
        
        # check if embeddings exist
        if False and not os.path.exists(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_observation.pkl'))):

            observations = []
            date_times = []
            context_ids = []
            for i in range(1,50):

                if 'session_%s' % i not in data:
                    break
                if len(data['session_%s' % i]) == 0:
                   continue    

                if 'session_%s_observation' % i not in data:
                    print("Getting observations for session %s" % i)
                    facts = get_session_facts(args, data, data, i)
                    data['session_%s_observation' % i] = facts
                    with open(ann_file, 'w') as f:
                        json.dump(data, f, indent=2)
                else:
                    facts = data['session_%s_observation' % i]

                date_time = data['session_%s_date_time' % i]
                for k, v in facts.items():
                    for fact, dia_id in v:
                        observations.append(fact)
                        context_ids.append(dia_id)
                        date_times.append(date_time)

            print("Getting embeddings for %s observations" % len(observations))
            embeddings = get_embeddings(args, observations, 'context')
            print(embeddings.shape)
            database = {'embeddings': embeddings,
                             'date_time': date_times,
                             'dia_id': context_ids,
                             'context': observations}

            with open(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_observation.pkl')), 'wb') as f:
                pickle.dump(database, f)

        else:
            database = pickle.load(open(os.path.join(args.emb_dir, os.path.split(ann_file)[-1].replace('.json', '_observation.pkl')), 'rb'))
    
    print("Getting embeddings for %s questions" % len(data['qa']))
    question_embeddings = get_embeddings(args, [q['question'] for q in data['qa']], 'query')

    return database, question_embeddings


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
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--retriever', type=str, default="contriever")
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


def get_rag_context(context_database, query_vector, args):

    output = np.dot(query_vector, context_database['embeddings'].T)
    sorted_outputs = np.argsort(output)[::-1]
    sorted_context = [context_database['context'][idx] for idx in sorted_outputs[:args.top_k]]
    
    sorted_context_ids = []
    for idx in sorted_outputs[:args.top_k]:
        context_id = context_database['dia_id'][idx]
        if type(context_id) == str:
            if ',' in context_id:
                context_id = [s.strip() for s in context_id.split(',')]
        if type(context_id) == list:
            sorted_context_ids.extend(context_id)
        else:
            sorted_context_ids.append(context_id)

    # sorted_context_ids = [context_database['dia_id'][idx] for idx in sorted_outputs[:args.top_k]]
    sorted_date_times = [context_database['date_time'][idx] for idx in sorted_outputs[:args.top_k]]
    if args.rag_mode in ['dialog', 'observation']:
        query_context = '\n'.join([date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])
    else:
        query_context = '\n\n'.join([date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])

    return query_context, sorted_context_ids


def get_input_context(data, num_question_tokens, encoding, args):

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
                    turn = dialog['speaker'] + ' said, \"' + dialog['compressed_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['compressed_text'] + '\n'
                except KeyError:
                    turn = dialog['speaker'] + ' said, \"' + dialog['clean_text'] + '\"' + '\n'
                    # conv = conv + dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'
                if "img_file" in dialog and len(dialog["img_file"]) > 0:
                    turn += '[shares %s]\n' % dialog["blip_caption"]
        
                num_tokens = len(encoding.encode('DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + turn))
                if (num_tokens + len(encoding.encode(query_conv)) + num_question_tokens) < (MAX_LENGTH[args.model]-(PER_QA_TOKEN_BUDGET*(args.batch_size))): # 20 tokens assigned for answers
                    query_conv = turn + query_conv
                else:
                    min_session = i
                    stop = True
                    break
            query_conv = 'DATE: ' + data['session_%s_date_time' % i] + '\n' + 'CONVERSATION:\n' + query_conv
        if stop:
            break
        
        if min_session == -1:
            print("Saved %s tokens in query conversation from full conversation" % len(encoding.encode(query_conv)))
        else:
            print("Saved %s conv. tokens + %s question tokens in query from %s out of %s sessions" % (len(encoding.encode(query_conv)), num_question_tokens, max_session-min_session, max_session))

    return query_conv


def get_gpt_answers(ann_file, out_file, args):


    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-16k' if any([k in args.model for k in ['16k', '12k', '8k', '4k']]) else args.model)
    in_data = json.load(open(ann_file))


    # start instruction prompt
    speakers_names = list(set([d['speaker'] for d in in_data['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = len(encoding.encode(start_prompt))

    if os.path.exists(out_file):
        out_data = json.load(open(out_file))
    else:
        out_data = in_data.copy()

    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))

    # start instruction prompt
    speakers_names = list(set([d['speaker'] for d in in_data['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = len(encoding.encode(start_prompt))

    if args.rag_mode:
        assert args.batch_size == 1, "Batch size need to be 1 for RAG-based evaluation."
        context_database, query_vectors = prepare_for_rag(args, in_data, ann_file)
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


        if args.use_rag:
            query_conv, context_ids = get_rag_context(context_database, query_vectors[include_idxs][0], args) # rag mode is set to batch size 1
        else:
            question_prompt =  QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            num_question_tokens = len(encoding.encode(question_prompt))
            query_conv = get_input_context(in_data, num_question_tokens + start_tokens, encoding, args)
            query_conv = start_prompt + query_conv
        

        print("%s tokens in query" % len(encoding.encode(query_conv)))

        if 'gpt-4' in args.model:
            time.sleep(5)

        elif args.model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-12k', 'gpt-3.5-turbo-8k', 'gpt-3.5-turbo-4k']:
            time.sleep(1)

        if args.batch_size == 1:

            query = query_conv + '\n\n' + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + QA_PROMPT_CAT_5.format(questions[0])
            print('------------------------------------------------------------------------------------')
            print(query)
            print('------------------------------------------------------------------------------------')
            answer = run_chatgpt(query, num_gen=1, num_tokens_request=32, 
                    model='chatgpt' if 'gpt-3.5' in args.model else args.model, 
                    use_16k=True if any([k in args.model for k in ['16k', '12k', '8k', '4k']]) else False, 
                    temperature=0, wait_time=2)
            
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
                    print("Sending query of %s tokens" % len(encoding.encode(query)))
                    print("Trying with answer token budget = %s per question" % PER_QA_TOKEN_BUDGET)
                    answer = run_chatgpt(query, num_gen=1, num_tokens_request=args.batch_size*PER_QA_TOKEN_BUDGET, 
                            model='chatgpt' if 'gpt-3.5' in args.model else args.model, 
                            use_16k=True if any([k in args.model for k in ['16k', '12k', '8k', '4k']]) else False, 
                            temperature=0, wait_time=2)
                    answer = answer.replace('\\"', "'").replace('json','').replace('`','').strip().replace("\\'", "")
                    print(answer)
                    # try:
                    #     answers = json.loads(answer.strip())
                    # except:
                    answers = process_ouput(answer.strip())
                    break

                except Exception as e:
                    print(e)
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
    set_openai_key()

    # output directory
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.json') if '26' in f] # fix for other files
    # data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if 'Chat_8' in f] # fix for other files
    data_files = [os.path.join(args.data_dir, f) for f in ['41_post_qa_post_clean_adv.json']]
    # data_files = [os.path.join(args.data_dir, f) for f in ['42_post_qa_post_clean_adv.json',
    #                                                        '43_post_qa_post_clean_adv_new.json', 
    #                                                        '44_post_qa_post_clean_adv_new.json',
    #                                                        '47_post_qa_post_clean_adv_new.json',
    #                                                        '48_post_qa_post_clean_adv_new.json',
    #                                                        '49_post_qa_post_clean_adv_new.json',
    #                                                        '50_post_qa_post_clean_adv_new.json',
    #                                                        '26_post_qa.json', 
    #                                                        '30_post_qa.json']]

    prediction_key = "%s_prediction" % args.model if not args.use_rag else "%s_%s_top_%s_prediction" % (args.model, args.rag_mode, args.top_k)

    ems = []
    total = 0
    for f in data_files:
        get_gpt_answers(f, os.path.join(args.out_dir, os.path.split(f)[-1]), args)
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

