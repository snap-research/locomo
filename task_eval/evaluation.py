import regex
import json
import string
import unicodedata
from typing import List
import numpy as np
from collections import Counter
from bert_score import score
from nltk.stem import PorterStemmer
ps = PorterStemmer()

LENGTH_THRESHOLD = 5

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def check_answer(example, tokenizer) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers = example['answers']
    ctxs = example['ctxs']

    hits = []

    for _, doc in enumerate(ctxs):
        text = doc['text']

        if text is None:  # cannot find the document for some reason
            hits.append(False)
            continue

        hits.append(has_answer(answers, text, tokenizer))

    return hits


def has_answer(answers, text, tokenizer=SimpleTokenizer()) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


def _normalize(text):
    return unicodedata.normalize('NFD', text)


def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):

    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    # print('# EM #', prediction, ' | ', ground_truth, ' #', set(prediction.split()) == set(ground_truth.split()))
    # return normalize_answer(prediction) == normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())
    
# def bert_score(prediction, ground_truths):
#     prediction = normalize_answer(prediction)
#     values = []
#     for ground_truth in ground_truths:
#         ground_truth = normalize_answer(ground_truth)
#         P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
#         values.append(R[0].item())
#     print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
#     return max(0, max(values))


def bert_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    P, R, F1 = score([prediction], [ground_truth], lang='en', verbose=False, rescale_with_baseline=True)
    # print('# BERT # ', normalize_answer(prediction), ' | ', normalize_answer(ground_truth), ' #', P, R, F1)
    return max(0, F1[0].item())


def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1


def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])


def rougel_score(prediction, ground_truth):
    from rouge import Rouge
    rouge = Rouge()
    prediction = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
    ground_truth = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])


## file-level evaluation ... ### 
def eval_recall(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    has_answer_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = ' || '.join(line['output'])

        if has_answer(answer, output, tokenizer):
            has_answer_count += 1

        answer_lengths.append(len(output.split()))

    recall = round(has_answer_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return recall, lens


def eval_question_answering(infile, eval_key='prediction', metric='f1'):

    # lines = open(infile, 'r').readlines()[1:]
    lines = json.load(open(infile, 'r'))['qa']

    all_ems = []
    all_recall = []
    exact_match_count = 0
    f1_count = 0
    answer_lengths = []
    for i, line in enumerate(lines):
        # line = json.loads(line)
        if type(line[eval_key]) == list:
            answer = line['answer']
        else:
            answer = str(line['answer'])
        if line['category'] == 3:
            answer = answer.split(';')[0].strip()
        
        output = line[eval_key]

        if "metric" in line and line["metric"] == "bertscore":
            # print(answer, output)
            all_ems.append(bert_score(output, answer))
            answer_lengths.append(len(output.split()))
        
        elif line['category'] in [2, 3, 4]:
            # print(answer, output)
            all_ems.append(f1_score(output, answer))
            # all_ems.append(rougel_score(output, answer))
            # all_ems.append(bert_score(output, answer))
        
        elif line['category'] in [1]:
            # all_ems.append(bert_score(output, answer))
            all_ems.append(f1(output, answer))
            # all_ems.append(rougel_score(output, answer))
            answer_lengths.append(len(output.split()))

        elif line['category'] in [5]:
            # all_ems.append(1-rougel_score(output, answer))
            if 'no information available' in output.lower() or 'not mentioned' in output.lower():
                all_ems.append(1)
            else:
                all_ems.append(0)
                # all_ems.append(1-f1_score(output, answer))

        else:
            print(line)
            raise ValueError
        
        assert i+1 == len(all_ems), all_ems

        if eval_key + '_context' in line and len(line['evidence']) > 0:
            # recall_acc for dialog
            if line[eval_key + '_context'][0].startswith('S'):
                sessions = [e[1:] for e in line[eval_key + '_context']]
                recall_acc = float(sum([ev.split(':')[0][1:] in sessions for ev in line["evidence"]]))/len(line['evidence'])
            else:
                recall_acc = float(sum([ev in line[eval_key + '_context'] for ev in line["evidence"]]))/len(line['evidence'])
            all_recall.append(recall_acc)
        else:
            all_recall.append(1)

    print("{} QA samples evaluated; {} accuracy values".format(len(lines), len(all_ems)))
    lens = round(np.mean(answer_lengths), 4)
    return all_ems, lens, all_recall


def eval_fact_checking(infile):

    tokenizer = SimpleTokenizer()
    lines = open(infile, 'r').readlines()[1:]

    exact_match_count = 0
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        if answer == ["refutes"]:
            answer = ["refutes", "no", "false"]
        if answer == ["supports"]:
            answer = ["supports", "yes", "true"]

        if has_answer(answer, output, tokenizer):
            exact_match_count += 1
        
        answer_lengths.append(len(output.split()))

    em = round(exact_match_count/len(lines), 4)
    lens = round(np.mean(answer_lengths), 4)

    return em, lens


def eval_dialogue_system(infile):

    lines = open(infile, 'r').readlines()[1:]

    f1_scores = []
    rl_scores = []
    answer_lengths = []
    for line in lines:
        line = json.loads(line)
        answer = line['answer']
        output = line['output'][0]

        f1_scores.append(f1(output, answer))
        rl_scores.append(rl(output, answer))
        answer_lengths.append(len(output.split()))

    F1 = round(np.mean(f1_scores), 4)
    RL = round(np.mean(rl_scores), 4)
    lens = round(np.mean(answer_lengths), 4)

    return F1, RL, lens