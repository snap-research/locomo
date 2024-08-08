from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]


messages = [
    {"role": "user", "content": "You are an intelligent assistant that can read conversations happening between two people and then answer questions based on the conversation."},
    {"role": "assistant", "content": "I am an intelligent assistant that can read conversations happening between two people and then answer questions based on the conversation."},
    {"role": "user", "content": "Here is a conversation between two people. %s Answer with exact words from the conversation. What is Tiffany Horning's favorite postcard?" % open('./plots_and_txts/3.txt').read()[-40000:]}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=50, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])