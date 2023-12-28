from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

def tokenize_fn(example):
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    tokenizer.add_eos_token=True
    return tokenizer(example)

def format_and_tokenize_examples_from_json(example):
    formatted_and_tokenized_examples = []
    for item in example:
        formatted_string = f"[INST] Question: {item['question']} [/INST] Answer: {item['answer']}"
        tokenized = tokenize_fn(formatted_string)
        formatted_and_tokenized_examples.append(tokenized)
    return formatted_and_tokenized_examples



example_json = [
        {
            "context":"The sky is blue.",
            "question":"What color is the sky?",
            "answer":"Blue."
        },
        {
            "context":"When you add 2+2 it equals 4.",
            "question":"What is 2+2?",
            "answer":"4."
        },
    ]

print(format_and_tokenize_examples_from_json(example_json))
