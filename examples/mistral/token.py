from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

training_example = "Question: What color is the sky? \n\n Answer: Blue."

print(tokenizer)
