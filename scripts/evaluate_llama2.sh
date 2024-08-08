# python3 task_eval/evaluate_llama2.py \
#     --data-dir ./data/multimodal_dialog/final/ \
#     --out-dir ./outputs/all/ --model mistral-instruct-7b-8k-new --use-4bit --overwrite

python3 task_eval/evaluate_llama2.py \
    --data-dir ./data/multimodal_dialog/final/ \
    --out-dir ./outputs/all/ --model mistral-instruct-7b-32k-v2 --use-4bit

python3 task_eval/evaluate_llama2.py \
    --data-dir ./data/multimodal_dialog/final/ \
    --out-dir ./outputs/all/ --model llama2-chat-70b --use-4bit

python3 task_eval/evaluate_llama2.py \
    --data-dir ./data/multimodal_dialog/final/ \
    --out-dir ./outputs/all/ --model llama3-chat-70b --use-4bit

# python3 task_eval/evaluate_llama2.py \
#     --data-dir ./data/multimodal_dialog/final/ \
#     --out-dir ./outputs/all/ --model gemma-7b-it --use-4bit