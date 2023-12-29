from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

def tokenize_fn(example):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    tokenizer.add_eos_token=True
    return tokenizer(example)

example = "Question: What color is the sky?  Answer: Blue."
example_json = [
        {
            "context":"The sky is blue",
            "question":"What color is the sky?",
            "answer":"Blue"
        },
        {
            "context":"When you add 2+2 it equals 4.",
            "question":"What is 2+2?",
            "answer":"4."
        },
    ]

#print(tokenizer([example]))
#print(tokenize_fn(example))
#
#print(tokenizer.add_eos_token)
#tokenizer.add_eos_token=True
#print(tokenizer.add_eos_token)
#
#print("---------")

formatted_examples = []
for item in example_json:
    formatted_string = f"[INST] Question: {item['question']} [/INST] Answer: {item['answer']}."
    formatted_examples.append(formatted_string)

for example in formatted_examples:
    tokenized = tokenize_fn(example)
    converted = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
    print(converted)

print("----")
def tokenize_inst():
    inst_example = "[INST]"
    inst = tokenize_fn(inst_example)
    inst = tokenizer.convert_ids_to_tokens(inst['input_ids'])
    print(f"tokenized inst: {inst}")

tokenize_inst()



