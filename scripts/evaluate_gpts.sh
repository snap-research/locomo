# LoCoMo

# python3 qa_eval/evaluate_lms.py \
#     --data-dir ./data/multimodal_dialog/data/ \
#     --out-dir ./outputs/chatgpt/ \
#     --openai-key-file ./keys/openai.key.txt \
#     --model chatgpt


# python3 task_eval/evaluate_gpts.py \
#     --data-dir ./data/multimodal_dialog/final/ \
#     --out-dir ./outputs/all/ \
#     --openai-key-file ./keys/openai.key.txt \
#     --model gpt-3.5-turbo-4k \
#     --batch-size 10


# python3 task_eval/evaluate_gpts.py \
#     --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ --openai-key-file ./keys/openai.key.txt \
#     --model gpt-3.5-turbo --batch-size 1 --use-rag --retriever dragon --top-k 2 \
#     --emb-dir ./outputs/all/ --rag-mode summary --prompt-dir ./prompt_examples

python3 task_eval/evaluate_gpts.py \
    --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
    --model gpt-4-turbo --batch-size 20 \
    --emb-dir ./outputs/all/ --prompt-dir ./prompt_examples

python3 task_eval/evaluate_gpts.py \
    --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
    --model gpt-3.5-turbo-4k --batch-size 10 \
    --emb-dir ./outputs/all/ --prompt-dir ./prompt_examples

python3 task_eval/evaluate_gpts.py \
    --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
    --model gpt-3.5-turbo-8k --batch-size 10 \
    --emb-dir ./outputs/all/ --prompt-dir ./prompt_examples

python3 task_eval/evaluate_gpts.py \
    --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
    --model gpt-3.5-turbo-12k --batch-size 10 \
    --emb-dir ./outputs/all/ --prompt-dir ./prompt_examples

python3 task_eval/evaluate_gpts.py \
    --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
    --model gpt-3.5-turbo-16k --batch-size 10 \
    --emb-dir ./outputs/all/ --prompt-dir ./prompt_examples


# python3 task_eval/evaluate_gpts.py \
#     --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ --openai-key-file ./keys/openai.key.txt \
#     --model gpt-3.5-turbo --batch-size 1 --use-rag --retriever dragon --top-k 2 \
#     --emb-dir ./outputs/all/ --rag-mode summary --prompt-dir ./prompt_examples


# python3 task_eval/evaluate_gpts.py \
#     --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
#     --model gpt-3.5-turbo --batch-size 1 --use-rag --retriever dragon --top-k 2 \
#     --emb-dir ./outputs/all/ --rag-mode summary --prompt-dir ./prompt_examples

# python3 task_eval/evaluate_gpts.py \
#     --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
#     --model gpt-3.5-turbo --batch-size 1 --use-rag --retriever dragon --top-k 5 \
#     --emb-dir ./outputs/all/ --rag-mode summary --prompt-dir ./prompt_examples

# python3 task_eval/evaluate_gpts.py \
#     --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
#     --model gpt-3.5-turbo --batch-size 1 --use-rag --retriever dragon --top-k 10 \
#     --emb-dir ./outputs/all/ --rag-mode summary --prompt-dir ./prompt_examples

# python3 task_eval/evaluate_gpts.py \
#     --data-dir ./data/multimodal_dialog/final/ --out-dir ./outputs/all/ \
#     --model gpt-3.5-turbo --batch-size 1 --use-rag --retriever dragon --top-k 50 \
#     --emb-dir ./outputs/all/ --rag-mode dialog --prompt-dir ./prompt_examples

# LoCoMo Real

# python3 task_eval/evaluate_gpts.py \
#     --data-dir ./data/multimodal_dialog/quest_data_final/with_qa/ \
#     --out-dir ./data/multimodal_dialog/quest_data_final/qa_outputs/ --openai-key-file ./keys/openai.key.txt \
#     --model gpt-3.5-turbo-16k --batch-size 10 --prompt-dir ./prompt_examples