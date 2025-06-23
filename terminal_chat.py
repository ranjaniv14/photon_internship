import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chat.chat import chat_with_llm

def main():
    print("💬 Terminal Chatbot (via Ollama API). Type 'exit' to quit.\n")
    history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            break
        reply, history = chat_with_llm(user_input, history)
        print(f"Assistant: {reply}\n")

if __name__ == "__main__":
    main()
