import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import argparse
import os, json
from generative_agents.memory_utils import get_session_facts
from global_methods import set_openai_key
from task_eval.dpr_qa import get_embeddings
import pickle

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out-file', type=str, required=True)
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--emb-dir', type=str, default="")
    parser.add_argument('--prompt-dir', type=str, default="")
    parser.add_argument('--use-date', action="store_true")
    parser.add_argument('--overwrite', action="store_true", help="set flag to overwrite existing outputs")
    parser.add_argument('--retriever', type=str, default="dragon")

    args = parser.parse_args()
    return args


def main():

    
    # set openai API key
    set_openai_key()

    # get arguments
    args = parse_args()

    # load conversations
    samples = json.load(open(args.data_file))

    # load the output file if it exists to check for overwriting
    if os.path.exists(args.out_file):
        out_samples = {d['sample_id']: d for d in json.load(open(args.out_file))}
    else:
        out_samples = {}

    for data in samples:

        observations = []
        date_times = []
        context_ids = []

        # check for existing output
        if data['sample_id'] in out_samples:
            output = out_samples['sample_id']
        else:
            output = {'sample_id': data['sample_id']}

        session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if 'date_time' not in k]
        for i in tqdm(range(min(session_nums), max(session_nums) + 1), desc='Generating observations for %s' % data['sample_id']):

            if 'session_%s_observation' % i not in output or args.overwrite:
                facts = get_session_facts(args, data, data, i, return_embeddings=False)
                data['session_%s_observation' % i] = facts
            else:
                facts = data['session_%s_observation' % i]

            date_time = data['session_%s_date_time' % i]
            for k, v in facts.items():
                for fact, dia_id in v:
                    observations.append(fact)
                    context_ids.append(dia_id)
                    date_times.append(date_time)

        if args.use_date:
            inputs = ['. '.join([k, v]) for k, v in zip(date_times, observations)]
            embeddings = get_embeddings(args, inputs, 'context')
        else:
            embeddings = get_embeddings(args, observations, 'context')
        
        assert embeddings.shape[0] == len(observations)

        database = {'embeddings': embeddings,
                            'date_time': date_times,
                            'dia_id': context_ids,
                            'context': observations}

        with open(args.out_file.replace('.json', '_%s_obs_emb.pkl' % data['sample_id']), 'wb') as f:
            pickle.dump(database, f)


main()