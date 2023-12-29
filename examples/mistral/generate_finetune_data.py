import os
import json
import requests
import fitz
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv
from urls import urls

load_dotenv()
openai_client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY")
)

replacements = {
    "ﬁ": "fi",
    "ﬂ": "fl",
}

def parse_pdf(file_path):
    doc = fitz.open(file_path)
    chunks = []

    for page in doc:
        blocks = page.get_text("blocks")

        for block in blocks:
            text = block[4]

            for key, value in replacements.items():
                text = text.replace(key, value)

            if len(text) >= 50:
                chunks.append(text)

    doc.close()
    return chunks

def parse_pdf_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
    chunks = []
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4]
            for key, value in replacements.items():
                text = text.replace(key,value)
            if len(text) >= 100 and '.edu' not in text and '.com' not in text:
                text = text.replace('\n', ' ')
                chunks.append(text)

    doc.close()
    return chunks


def get_summary(chunk):
    system_prompt="You are a helpful assistant. Please provide summaries that are concise but don't leave out important information to the text."

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please summarize this text while keeping key information: {chunk}"}
        ]
    )
    response = completion.choices[0].message.content

    return response

def get_question(chunk):
    system_prompt="You are a helpful assistant tasked with coming up with some quiz questions, for college students. Please come up with a concise question that can be answered with the following information. The questions will be used to quiz students knowledge on the concepts."

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"We are coming up with a quiz to test college students. Come up with a quiz question that is relevant and can be answered with the following information: {chunk}"}
        ]
    )
    response = completion.choices[0].message.content

    return response
    
def get_answer(question,chunk):
    system_prompt="You are a helpful assistant. Please help users by answering their questions given the provided context. Use the information specifically from the context documents to answer the question. Your response should be concise while still answering any details of the question."

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"The user question: {question} \n\n The context documents: {chunk}"}
        ]
    )
    response = completion.choices[0].message.content

    return response

def get_answer_no_context(question):
    system_prompt="You are a helpful assistant."

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": {question}}
        ]
    )
    response = completion.choices[0].message.content

    return response



#examples = {
#    "questions": [
#        "What arithmetic is typically used for DNN inference?",
#        "What kind of computation is fundamental for a DNN?"
#    ],
#    "context": [
#        "Traditionally 32-bit floating point arithmetic is used for DNN inference. However, the IEEE standard floating point"
#        " representation is designed for a very broad dynamic range; even 32-bit floating point numbers have a huge dynamic"
#        " range of over 80 orders of magnitude, far larger than needed for DNNs. While very small values can be important, very"
#        " large values are not, therefore the design of the numbers creates low information-per-bit based on Shannon maximum entropy",
#        "The fundamental computation within a DNN is the multiplyand-accumulate (MAC) operation. Each neuron within a network is equivalent to a MAC unit in that it performs a weighted"
#        " sum of its inputs. This operation is ubiquitous across many DNN implementations, however, the operation is usually inexact, i.e. limited precision, truncation, or premature rounding in"
#        " the underlying hardware yields inaccurate results. The EMAC performs the same computation but allocates sufficient padding for digital signals to emulate arbitrary precision. Rounding or"
#        " truncation within an EMAC unit is delayed until every product has been accumulated, thus producing a result with minimal local error. This minimization of error is especially important"
#        " when EMAC units are coupled with low-precision data."
#    ]
#}




def chunk_summary_question_answer(file_path, chunk_index):
    chunks = parse_pdf(file_path)

    if chunks and chunk_index < len(chunks):
        chunk = chunks[chunk_index]
        question = get_question(chunk)
        answer = get_answer(question, chunk)

        print(f"Chunk: {chunk}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    else:
        print("No suitable chunk found at the specified index.")

def chunk_summary_question_answer_url(url, chunk_index):
    chunks = parse_pdf_from_url(url)

    if chunks and chunk_index < len(chunks):
        chunk = chunks[chunk_index]
        question = get_question(chunk)
        answer = get_answer(question, chunk)

        print(f"Chunk: {chunk}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
    else:
        print("No suitable chunk found at the specified index.")



#file_path="1812.01762.pdf"
#url=r"https://arxiv.org/pdf/1812.01762.pdf"
#chunk_index = 17
#print(len(parse_pdf(file_path)))
#chunk_summary_question_answer(file_path, chunk_index)
#chunk_summary_question_answer_url(url, chunk_index)





#chunks = parse_pdf(file_path)
#if chunks:
#    for i in range(15, 20):
#        print(chunks[i])
#else:
#    print("No chunks found")

#if chunks:
#    chunk = chunks[16]
#    summary = get_summary(chunk)
#    print(f"Chunk: {chunks[16]}")
#    print(f"Summary: {summary}")
#else:
#    print("No suitable chunks found in the document.")


def format_for_finetuning(question_answer_pairs):
    formatted_samples = []
    for question, answer in question_answer_pairs:
        formatted_sample = f"[INST] Question: {question} [/INST] Answer: {answer}" #f"<s>[INST] {question} [/INST] {answer}</s>"
        formatted_samples.append(formatted_sample)

    return formatted_samples


def main(urls):
    formatted_pairs = []
    for i, url in enumerate(urls[:2]):
        print(f"Parsing {i+1}/{len(urls[:2])} Documents")
        chunks = parse_pdf_from_url(url)
        for chunk in chunks[10:15]:
            question = get_question(chunk)
            answer = get_answer(question, chunk)
            formatted_pairs.append({"chunk": chunk, "question": question, "answer": answer})
        print(f"{len(chunks[10:15])} out of {len(chunks)} chunks processed.")

    with open('qa_pairs.json', 'w') as f:
        json.dump(formatted_pairs, f)

    return formatted_pairs


if __name__ == "__main__":
    main(urls)
