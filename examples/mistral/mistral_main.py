from transformers import AutoTokenizer
from urls import urls
from generate_finetune_data import *

def generate_qa_examples():
    formatted_pairs = []
    for url in urls:
        chunks = parse_pdf_from_url(url)
        for chunk in chunks:
            question = get_question(chunk)
            answer = get_answer(question, chunk)
            formatted_pairs = formated_for_finetuning([(question, answer)])

    return formatted_pairs

