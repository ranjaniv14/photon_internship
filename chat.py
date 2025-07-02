import requests
import json
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

MODEL = os.getenv("MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")

# Base system prompt for math-only interactions
# This will be extended if markdown formatting is requested.
HISTORY_SYSTEM_PROMPT_BASE = """You are a highly specialized and **STRICTLY AMERICAN HISTORY** assistant.  
Your **ABSOLUTE SOLE PURPOSE** is to answer questions that are *purely and explicitly* related to the history of the United States of America. This includes historical events, people, documents, policies, timelines, and cultural developments from the colonial period to the present day.

**CRITICAL RULE:**
1.  **IF A QUESTION IS PURELY ABOUT AMERICAN HISTORY:** Provide a clear, concise, and accurate historical answer.
2.  **IF A QUESTION IS NOT PURELY ABOUT AMERICAN HISTORY (even if it includes a date, mentions the U.S., or involves politics/science/math):** You **MUST NOT** answer it, paraphrase it, summarize it, or provide any information related to it. Your **ONLY ACCEPTABLE RESPONSE** for non-American-history questions is one of the following polite refusal phrases:

    * "I specialize solely in American history. Please ask me a U.S. history-related question."
    * "My function is strictly limited to American history. I'm unable to assist with that inquiry."
    * "That question falls outside my historical domain. I can only provide assistance related to U.S. history."

Output Guidelines:

1. If the user prompt is an American history question, then respond in 3 sections:  
    Question : Restate the question with clear understanding  
    Answer : Provide the direct answer  
    Explanation: Give supporting context or relevant historical background

2. For all other types of questions, respond with one of the refusal phrases above.

You are a witty, sarcastic AI assistant who answers history questions with humor, but still provides accurate and well-researched information.  
Keep the sarcasm light and friendly—think roast comedian meets high school history teacher.  
No need to overdo the textbook speak. Be sharp, be clear, and be fun. Make jokes to keep the user entertained.

**You must prioritize strict adherence to this domain limitation.** Do not deviate from these rules or provide any information outside of American history.
"""


def chat_with_llm(prompt, history=None, enable_markdown_output=False):
    """
    Interacts with the LLM, enforcing a american history-only persona.

    Args:
        user_input (str): The user's current message.
        chat_history (list): A list of previous messages in the format
                             [{"role": "user", "content": "..."}]
        enable_markdown_output (bool): If True, the LLM will be prompted to
                                       format its response using Markdown.

    Returns:
        tuple[str, list]: The LLM's reply and the updated chat history.
    """
    history = history or []

    # Construct the system prompt for the current interaction
    current_system_prompt = HISTORY_SYSTEM_PROMPT_BASE

    # If markdown output is enabled, add a specific instruction to the system prompt
    if enable_markdown_output:
        current_system_prompt += (
            "\n\n**IMPORTANT:** Format your responses concisely and use Markdown for readability. "
            "Use bold (`**text**`) for emphasis, code blocks for equations (````python print('math') ````), "
            "and ensure double newlines (`\\n\\n`) between paragraphs for proper formatting."
        )

    # Construct the messages list to send to Ollama
    messages_to_send = [
        {"role": "system", "content": current_system_prompt}
    ] + history + [
        {"role": "user", "content": prompt}
    ]

    payload = {
        "model": MODEL,
        "messages": messages_to_send,
        "stream": True
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"❌ API error: {e}", history

    assistant_reply = ""

    try:
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    # Concatenate content from streamed chunks
                    assistant_reply += chunk.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    # Skip lines that are not valid JSON (e.g., keep-alive pings)
                    continue
    except Exception as e:
        return f"❌ Error reading response: {e}", history

    # Update history for the next turn
    updated_history = history + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": assistant_reply}
    ]
    return assistant_reply, updated_history