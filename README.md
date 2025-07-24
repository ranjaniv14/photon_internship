# photon_internship
# 🦙 LLM Terminal Chatbot (via Ollama API)

This project provides an interactive terminal-based chatbot powered by [TinyLlama](https://ollama.com/library/tinyllama), running locally using [Ollama](https://ollama.com) and accessed through Python's HTTP API.

---

## 📁 Repository Structure

GitHub repository: [ranjaniv14/photon_internship](https://github.com/ranjaniv14/photon_internship)

**Folder structure:**

```
photon_internship/
│
├── chat/
│   └── chat.py                ← Shared chat logic
│
├── terminal_chat/
│   └── terminal_chat.py       ← Terminal chatbot with main()
│
├── streamlit_chatbot/
│   └── streamlit_app.py       ← Streamlit app
│
└── requirements.txt
```

## ✅ Requirements
- Python 3.7 or higher
- Ollama installed and running
- Visual Studio Code
- Python extension for VS Code
- requests Python package

## ⚙️ Setup Instructions

1. Clone the Repository

```
git clone https://github.com/ranjaniv14/photon_internship.git
cd photon_internship/terminal_chatbot
```
2. Create a Python Virtual Environment

```
python -m venv .venv
```

3. Activate the Virtual Environment

- Windows CMD:
```
.venv\Scripts\activate
```
- On Windows (PowerShell)

```
.\.venv\Scripts\Activate.ps1
```
- On macOS / Linux
```
source .venv/bin/activate
```
4. Select the Python Interpreter in VS Code
- Open VS Code.
- Press Ctrl+Shift+P (or Cmd+Shift+P on macOS).
- Choose Python: Select Interpreter.
- Select the .venv environment from the list.

5. Install Dependencies
- Install required Python packages using:
```
pip install -r requirements.txt
```

## 🤖 Download the Model with Ollama
Make sure Ollama is installed and running. Then pull the TinyLlama model: The model can be changed as per our need.
```
ollama pull tinyllama
```
## ▶️ Run the Chatbot via terminal

- Run the chatbot script:
```
python terminal_chat/terminal_chat.py

## ▶️ Run the Chatbot via Streamlit App

- Run the chatbot :
```
streamlit run streamlit_chat/streamlit_chat.py
```

extract text from pdfs: 
streamlit run pdf_extractor.py
streamlit run embed_pdf.py
streamlit run pdf_reader.py

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install streamlit pymupdf sentence-transformers psycopg2-binary sqlalchemy pgvector
