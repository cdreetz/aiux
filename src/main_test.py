from config.clients import OpenAIClient
from src.db.query_chroma import query


def ai(prompt):
    openai_client = OpenAIClient.get_instance()
    context = query(prompt)

    system_prompt="""
    You are a helpful assistant tasked with answering financial wellness questions.  
    You can refer to the provided documents to get additional information based on user prompts.
    Keep your answers concise and to the point.
    If questions are not related to finances wellness, reply SALSA
    """

    messages=[]
    messages.append({"role": "system", "content": f"{system_prompt}"})
    messages.append({"role": "user", "content": f"The user question:{prompt} \n The context:{context}"})


    completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    response = completion.choices[0].message.content

    messages.append({"role": "assistant", "content": response})

    return response

def main():
    print("Welcome to the iGrad Assistant. Type 'quit' to exit.")
    while True:
        user_input = input(">You: ")
        if user_input.lower() == 'quit':
            break
        response = ai(user_input)
        print(">Assistant: ", response)
        print("---------")
        

if __name__ == "__main__":
    main()