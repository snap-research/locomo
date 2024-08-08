source scripts/env.sh

python task_eval/get_facts.py --emb-dir $EMB_DIR --use-date --prompt-dir $PROMPT_DIR --data-file locomo10.json --out-file $OUT_DIR/locomo10_observations.json