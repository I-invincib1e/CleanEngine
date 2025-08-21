import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

def main():
    # Get OpenAI API key from environment variable or prompt user
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
        if not api_key:
            print("No API key provided. Exiting.")
            return
        os.environ["OPENAI_API_KEY"] = api_key

    # Initialize LLM and memory
    llm = ChatOpenAI(openai_api_key=api_key)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)

    print("LLM Chatbot (LangChain + OpenAI) - Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if not user_input:
            continue
        try:
            response = conversation.predict(input=user_input)
            print("Bot:", response)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
