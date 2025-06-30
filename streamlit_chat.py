import sys, os
import streamlit as st

# Check Streamlit version for st.chat_message
if not hasattr(st, 'chat_message'):
    st.error("This feature requires Streamlit 1.25.0 or newer. Please upgrade Streamlit (`pip install --upgrade streamlit`).")
    st.stop() # Stop execution if version is too old

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chat.chat import chat_with_llm

st.title("🧠 LLM Chatbot (via Ollama API)")

if "history" not in st.session_state:
    st.session_state.history = []
if "user_input_value" not in st.session_state: # New session state variable to control input field's value
    st.session_state.user_input_value = ""

def submit():
    # Get the user input from the widget's key, not from st.session_state.user_input directly
    # st.session_state.user_input will hold the value *after* the widget is rendered
    user_input = st.session_state.user_input_widget_key # Use the key from the chat_input/text_input

    if user_input:
        reply, updated_history = chat_with_llm(user_input, st.session_state.history, True)
        st.session_state.history = updated_history
        # Clear the input field by setting the dedicated session state variable
        st.session_state.user_input_value = "" # This is the crucial line to clear the input
        # Note: You should generally avoid setting st.session_state.user_input_widget_key directly here
        # as it was the source of the error. user_input_value is a helper.

# Display chat messages using st.chat_message
for i, message in enumerate(st.session_state.history):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    # Add a full-width separator after each assistant's response,
    # but not after the very last message.
    if message["role"] == "assistant" and i < len(st.session_state.history) - 1:
        st.divider() # Full-width horizontal line

# Use st.chat_input for the message input
# Crucial: The `value` argument of the widget should be set from session_state.user_input_value
# And the `key` is used to retrieve the current value of the input field.
# The on_submit callback will trigger when the user presses Enter or the send button.
st.chat_input(
    "Your message:",
    on_submit=submit,
    key="user_input_widget_key", # This key gets the *current* value of the widget
    # We use the separate st.session_state.user_input_value to control the widget's displayed content
    # However, st.chat_input does not have a `value` parameter to clear it programmatically after submit.
    # The common pattern for clearing st.chat_input is to just let it clear itself upon submission,
    # which it does by default when using on_submit.
    # So, we can simplify this.
)

# Removed the st.text_input line as st.chat_input is preferred for chat UIs