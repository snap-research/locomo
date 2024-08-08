# choices are dpr, contriever, dragon, openai
python3 task_eval/dpr_qa.py \
    --data-dir ./data/multimodal_dialog/final/ \
    --out-dir ./outputs/dragon/ \
    --openai-key-file ./keys/openai.key.txt \
    --retriever dragon --reader chatgpt