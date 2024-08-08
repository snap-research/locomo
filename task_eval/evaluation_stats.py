import os, json
import math
from tqdm import tqdm
from collections import defaultdict

def get_conversation_lengths(data, encoder=None):

    total_conv_length = 0
    id2length = {}
    for sess_num in range(1, 50):
        if 'session_%s' % sess_num not in data:
            continue
        if data['session_%s' % sess_num] == []:
            continue

        for dialog in data['session_%s' % sess_num]:
            try:
                dialog_tokens = dialog['speaker'] + ': ' + dialog['compressed_text'] + '\n'
            except KeyError:
                dialog_tokens = dialog['speaker'] + ': ' + dialog['clean_text'] + '\n'
            if "img_file" in dialog and len(dialog["img_file"]) > 0:
                dialog_tokens += '[shares %s]\n' % dialog["blip_caption"]
            if encoder is not None:
                dialog_length = len(encoder.encode(dialog_tokens))
            else:
                dialog_length = len(dialog_tokens.split())
            id2length[dialog["dia_id"]] = total_conv_length + dialog_length
            total_conv_length += dialog_length
    return id2length


def analyze_acc(ann_file, out_file, model_name, metric_key='f1', encoder=None, rag=False):

    total_counts = defaultdict(lambda: 0)
    acc_counts = defaultdict(lambda: 0)
    memory_counts = defaultdict(lambda: defaultdict(lambda: 0))
    memory_counts_og = defaultdict(lambda: defaultdict(lambda: 0))
    context_len_counts = defaultdict(lambda: 0)
    context_len_og = defaultdict(lambda: 0)
    data = json.load(open(ann_file))
    recall_by_category = defaultdict(lambda: 0)

    id2length = get_conversation_lengths(data, encoder)
    # print(id2length)

    for i, qa in tqdm(enumerate(data['qa'])):
        # if qa['category'] in [4, 5]:
        #     continue
        total_counts[qa['category']] += 1
        if metric_key in qa:
            acc_counts[qa['category']] += qa[metric_key]
            qa['evidence'] = [q.replace('(', '').replace(')', '') for q in qa["evidence"]]
            if len(qa['evidence']) > 0:
                # farthest_session = min([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                # memory_counts_og[farthest_session] += 1
                # if qa[metric_key]:
                #     memory_counts[farthest_session] += qa[metric_key]

                if rag:
                    recall_by_category[qa['category']] += qa[metric_key + '_recall']
                else:
                    try:
                        farthest_session = min([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                        farthest_dialog = min([int(e.split(':')[-1]) for e in qa['evidence'] if e != "" and int(e.split(':')[0][1:]) == farthest_session])
                        farthest_length = id2length['D' + str(farthest_session) + ':' + str(farthest_dialog)]
                        memory_counts_og[qa['category']][math.ceil(farthest_length/1000)] += 1
                        memory_counts[qa['category']][math.ceil(farthest_length/1000)] += qa[metric_key]

                        if qa['category'] == 1:
                            latest_session = max([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                            latest_dialog = max([int(e.split(':')[-1]) for e in qa['evidence'] if e != "" and int(e.split(':')[0][1:]) == latest_session])
                            latest_length = id2length['D' + str(latest_session) + ':' + str(latest_dialog)]
                            context_length = latest_length-farthest_length
                            context_len_og[math.ceil(context_length/1000)] += 1
                            context_len_counts[math.ceil(context_length/1000)] += qa[metric_key]
                    except:
                        continue
    
    print("Total number of questions and corresponding accuracy in each category: ")
    for k, v in total_counts.items():
        print(k, v, acc_counts[k], round(float(acc_counts[k])/v, 3))

    # print("Total number of questions and corresponding accuracy by memory")
    # keys = list(memory_counts_og.keys())
    # keys.sort()
    # results_by_memory = {"gpt3.5-16k": {}}
    # for k in keys:
    #     print(k, memory_counts_og[k], memory_counts[k], float(memory_counts[k])/memory_counts_og[k])
    #     results_by_memory["gpt3.5-16k"][k] = float(memory_counts[k])/memory_counts_og[k]
    
    if os.path.exists(out_file):
        results_dict = json.load(open(out_file))
    else:
        results_dict = {}

    results_dict[model_name] = {}
    results_dict[model_name]['category_counts'] = total_counts
    results_dict[model_name]['cum_accuracy_by_category'] = acc_counts

    if rag:
        results_dict[model_name]['recall_by_category'] = {k: v/total_counts[k] for k, v in recall_by_category.items()}
        print("Category and corresponding recall accuracy in each category: ")
        for k, v in recall_by_category.items():
            print(k, round(float(v)/total_counts[k], 3))
        print("Overall recall accuracy: ", sum(list(recall_by_category.values()))/sum(list(total_counts.values())))
    else:
        results_dict[model_name]['category_counts_by_memory'] = memory_counts_og
        results_dict[model_name]['cum_accuracy_by_category_by_memory'] = memory_counts
        results_dict[model_name]['context_length_counts'] = context_len_og
        results_dict[model_name]['cum_accuracy_by_context_length'] = context_len_counts

    with open(out_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    # for cats in [[1,5], [2,3,4]]:



    # for cat in [1,2,3,4,5]:

    #     memory_counts = defaultdict(lambda: 0)
    #     memory_counts_og = defaultdict(lambda: 0)
    #     for i, qa in tqdm(enumerate(data['qa'])):
    #         if qa['category'] != cat:
    #             continue
    #         if metric_key in qa:
    #             if len(qa['evidence']) > 0:
    #                 farthest_session = min([int(e.split(':')[0][1:]) for e in qa['evidence']])
    #                 memory_counts_og[farthest_session] += 1
    #                 if qa[metric_key] > 0:
    #                     memory_counts[farthest_session] += qa[metric_key]

    #     print("Total number of questions and corresponding accuracy by memory for category %s" % cat)
    #     keys = list(memory_counts_og.keys())
    #     keys.sort()
    #     for k in keys:
    #         print(k, memory_counts_og[k], memory_counts[k], float(memory_counts[k])/memory_counts_og[k])


def analyze_aggr_acc(data_dir, score_dir, out_file, model_name, metric_key, encoder=None, rag=False):

    total_counts = defaultdict(lambda: 0)
    acc_counts = defaultdict(lambda: 0)
    memory_counts = defaultdict(lambda: defaultdict(lambda: 0))
    memory_counts_og = defaultdict(lambda: defaultdict(lambda: 0))
    context_len_counts = defaultdict(lambda: 0)
    context_len_og = defaultdict(lambda: 0)
    recall_by_category = defaultdict(lambda: 0)

    # score_files = [f for f in os.listdir(score_dir) if f.endswith('_scores.json')]
    score_files = [f.replace('.json', '_scores.json') for f in ['41_post_qa_post_clean_adv.json', '42_post_qa_post_clean_adv.json',
                                                           '43_post_qa_post_clean_adv_new.json', 
                                                           '44_post_qa_post_clean_adv_new.json',
                                                           '47_post_qa_post_clean_adv_new.json',
                                                           '48_post_qa_post_clean_adv_new.json',
                                                           '49_post_qa_post_clean_adv_new.json',
                                                           '50_post_qa_post_clean_adv_new.json',
                                                           '26_post_qa.json', 
                                                           '30_post_qa.json']]
    
    for score_file in score_files:
        data = json.load(open(os.path.join(score_dir, score_file)))
        ann = json.load(open(os.path.join(data_dir, score_file.replace('_scores.json', '.json'))))
        id2length = get_conversation_lengths(ann, encoder)
        # print(id2length)

        for i, qa in tqdm(enumerate(data['qa'])):
            # if qa['category'] in [4, 5]:
            #     continue
            total_counts[qa['category']] += 1
            if metric_key in qa:
                
                acc_counts[qa['category']] += qa[metric_key]
                qa['evidence'] = [q.replace('(', '').replace(')', '') for q in qa["evidence"]]
                if len(qa['evidence']) > 0:
                    # farthest_session = min([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                    # memory_counts_og[farthest_session] += 1
                    # if qa[metric_key]:
                    #     memory_counts[farthest_session] += qa[metric_key]

                    if rag:
                        recall_by_category[qa['category']] += qa[metric_key + '_recall']
                    else:

                        try:
                            farthest_session = min([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                            farthest_dialog = min([int(e.split(':')[-1]) for e in qa['evidence'] if e != "" and int(e.split(':')[0][1:]) == farthest_session])

                            farthest_length = id2length['D' + str(farthest_session) + ':' + str(farthest_dialog)]


                            memory_counts_og[qa['category']][math.ceil(farthest_length/1000)] += 1
                            memory_counts[qa['category']][math.ceil(farthest_length/1000)] += qa[metric_key]

                            if qa['category'] == 1:
                                latest_session = max([int(e.split(':')[0][1:]) for e in qa['evidence'] if e != ""])
                                latest_dialog = max([int(e.split(':')[-1]) for e in qa['evidence'] if e != "" and int(e.split(':')[0][1:]) == latest_session])

                                latest_length = id2length['D' + str(latest_session) + ':' + str(latest_dialog)]
                                context_length = latest_length-farthest_length
                                context_len_og[math.ceil(context_length/1000)] += 1
                                context_len_counts[math.ceil(context_length/1000)] += qa[metric_key]
                        except:
                            continue
            else:
                print([k for k in qa.keys() if 'gemini' in k], metric_key)

    
    print("Total number of questions and corresponding accuracy in each category: ")
    total_k = 0
    total_v = 0
    # for k, v in total_counts.items():
    keys = [4, 1, 2, 3, 5]
    for k in keys:
        v = total_counts[k]
        print(k, v, acc_counts[k], round(float(acc_counts[k])/v, 3))
        total_v += acc_counts[k]
        total_k += v

    print("Overall accuracy: ", round(float(total_v)/total_k, 3))

    # print("Total number of questions and corresponding accuracy by memory")
    # keys = list(memory_counts_og.keys())
    # keys.sort()
    # results_by_memory = {"gpt3.5-16k": {}}
    # for k in keys:
    #     print(k, memory_counts_og[k], memory_counts[k], float(memory_counts[k])/memory_counts_og[k])
    #     results_by_memory["gpt3.5-16k"][k] = float(memory_counts[k])/memory_counts_og[k]
    
    if os.path.exists(out_file):
        results_dict = json.load(open(out_file))
    else:
        results_dict = {}

    results_dict[model_name] = {}
    results_dict[model_name]['category_counts'] = total_counts
    results_dict[model_name]['cum_accuracy_by_category'] = acc_counts

    if rag:
        results_dict[model_name]['recall_by_category'] = {k: v/total_counts[k] for k, v in recall_by_category.items()}
        print("Category and corresponding recall accuracy in each category: ")
        # for k, v in recall_by_category.items():
        keys = [4, 1, 2, 3, 5]
        for k in keys:
            v = recall_by_category[k]
            print(k, round(float(v)/total_counts[k], 3))
        print("Overall recall accuracy: ", sum(list(recall_by_category.values()))/sum(list(total_counts.values())))
    else:
        results_dict[model_name]['category_counts_by_memory'] = memory_counts_og
        results_dict[model_name]['cum_accuracy_by_category_by_memory'] = memory_counts
        results_dict[model_name]['context_length_counts'] = context_len_og
        results_dict[model_name]['cum_accuracy_by_context_length'] = context_len_counts

    with open(out_file, 'w') as f:
        json.dump(results_dict, f, indent=2)


if __name__ == "__main__":

    # analyze_acc('./data/multimodal_dialog/completed_annotations/3_out_gpt3.5_summary.json', 'gpt3.5-16k')
    
    # analyze_aggr_acc('./data/multimodal_dialog/quest_data_final/with_qa',
    #                  './data/multimodal_dialog/quest_data_final/qa_outputs', 
    #                  './data/multimodal_dialog/quest_data_final/qa_outputs/all_results.json',
    #                  'gpt-3.5-turbo',
    #                  'gpt-3.5-turbo_f1'
    #                  )
    
    # analyze_aggr_acc('./data/multimodal_dialog/quest_data_final/with_qa',
    #                 './data/multimodal_dialog/quest_data_final/qa_outputs', 
    #                 './data/multimodal_dialog/quest_data_final/qa_outputs/all_results.json',
    #                 'gpt-3.5-turbo-16k',
    #                 'gpt-3.5-turbo-16k_f1'
    #                 )

    # analyze_aggr_acc('./data/multimodal_dialog/final',
    #             './outputs/all', 
    #             './outputs/all_results.json',
    #             'gemini-pro-1.0',
    #             'gemini-pro-1.0_f1',
    #             rag=False
    #             )

    analyze_aggr_acc('./data/multimodal_dialog/final',
                './outputs/all', 
                './outputs/all_results.json',
                'llama3-chat-70b',
                'llama3-chat-70b_rouge',
                rag=False
                )

    # analyze_aggr_acc('./data/multimodal_dialog/final',
    #             './outputs/all', 
    #             './outputs/all_results.json',
    #             'gpt-3.5-turbo_summary_top_10',
    #             'gpt-3.5-turbo_summary_top_10_f1',
    #             rag=True
    #             )