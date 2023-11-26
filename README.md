# RAG-LLM
Conversational retrieval system implemented using the LangChain framework, using FAISS vectorstore. This project uses LLAMA-2 as LLM and instruct-xl as the Embedding Model. FastAPI is used to create the inference endpoint.

# Setup
Note: Please run code on GPU and Python 3.6+
1. Download the repository files<br>
2. Download llama2 model from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin<br>
3. Set paths for LLM and vectordb in .env file<br>
4. In command line write:
```bash
pip install -r requirements.txt
```
5. Run app
```bash
python app.py
```
The application will run on http://localhost:5000/ask_query endpoint.

# Sample Queries

    Query:
    ----------------
    Enter your query: Hello!
    
    System Response:
    -----------------
    Response: Hello! ask me anything.
    Query:
    ----------------
    Enter your query: Suggest some good hair products
    
    System Response:
    -----------------
    Response: Biotin & Collagen Volumizing Hair Shampoo + Biotin & Collagen Hair Conditioner, Argan-Liquid Gold Hair Spa, Cold Pressed Bhringraj Cooling Oil For Hair Fall & Damage Control
    Query:
    -----------------
    Enter your query: What is the market price of Salted Pumpkin?
    
    System Response:
    -----------------
    Response: The market price of Salted Pumpkin is 180.
    
    ======================================================================================================================================================================


   


