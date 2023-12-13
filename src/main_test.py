import uuid
import logging
from config.clients import OpenAIClient
from src.db.query_chroma import query
from config.logging_config import setup_logging

# Assuming query function is defined correctly and returns (context, metadata)

system_prompt = """
You are a helpful assistant tasked with answering user questions about our platform called iGrad.  
You can refer to the provided documents to get additional information based on user prompts.
Keep your answers concise and to the point.
Do not answer anything outside of the scope of the documents.
"""

def ai(prompt, messages):
    openai_client = OpenAIClient.get_instance()
    context, metadata = query(prompt)

    messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": f"The user question: {prompt} \n The context: {context}"})

    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content

    messages.append({"role": "assistant", "content": response})

    # Log the conversation with metadata
    logging.info({"role": "user", "content": prompt})
    logging.info({"context": metadata})
    logging.info({"role": "assistant", "content": response})

    return response

def main():
    messages = []
    conversation_id = str(uuid.uuid4())
    log_filename = f"logging/conversation_{conversation_id}.log"
    setup_logging(log_filename)
    
    logging.info("Starting application")
    messages.append({"role": "system", "content": system_prompt})
    logging.info({"role": "system", "content": system_prompt})

    print("Welcome to the iGrad Assistant. Type 'quit' to exit.")
    while True:
        user_input = input(">You: ")
        if user_input.lower() == 'quit':
            logging.info("Conversation ended by user")
            break
        response = ai(user_input, messages)
        print(">Assistant: ", response)
        print("---------")

if __name__ == "__main__":
    main()
