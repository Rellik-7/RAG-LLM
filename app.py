import os
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from pydantic import BaseModel
from langchain.callbacks import AsyncIteratorCallbackHandler
import sentence_transformers
from InstructorEmbedding import INSTRUCTOR
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()
load_dotenv()

DB_FAISS_PATH = os.getenv("VSTORE_PATH")
LLM_PATH = os.getenv("LLM_PATH")

#Loading the model
def load_llm():
    # Load the locally downloaded model here
    config = {'max_new_tokens': 1000, 'context_length':1000, 'repetition_penalty': 1.1}
    llm = CTransformers(
        model = LLM_PATH,
        model_type="llama",
        config=config,
        temperature = 0.5
    )
    return llm


loader = CSVLoader(file_path='./bigBasketProducts.csv', encoding="utf-8", csv_args={
            'delimiter': ','})
data = loader.load()

embeddings =  HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl',model_kwargs={"device":"cuda"})

db = FAISS.from_documents(data, embeddings)
db.save_local(DB_FAISS_PATH)
llm = load_llm()
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

def conversational_chat(query):
    result = chain({"question": query, "chat_history": []})
    return result["answer"]

class Item(BaseModel):
    text: str

@app.post("/ask_query")
def root(Data: Item):
    return {"Answer": conversational_chat(Data.text)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
    