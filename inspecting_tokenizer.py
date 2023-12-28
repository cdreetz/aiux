from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

training_example = "Question: What color is the sky? \n\n Answer: Blue."

tokenized = tokenizer([training_example])

#print(tokenizer)
#print(tokenized)
#dir(tokenizer)
print(tokenizer.bos_token)
print(tokenizer.add_bos_token)
print(tokenizer.eos_token)
print(tokenizer.add_eos_token)
tokenizer.add_eos_token = True
print(tokenizer.add_eos_token)


tokenized = tokenizer([training_example])
print(tokenized)

token_indices = tokenized['input_ids']
tokens = tokenizer.convert_ids_to_tokens(token_indices[0])
print(tokens)

print("----")

print(dir(tokenizer))

