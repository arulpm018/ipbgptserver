from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from transformers import AutoTokenizer, BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
import chromadb
import torch
from document_chat import chat_with_document, get_related_documents
from models import ThesisTitle, ChatQuery

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Authorization", "Content-Type"],
)

hf_key = 'HF_KEY'

embed_model = "/app/model/"
Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model, trust_remote_code=True)

db2 = chromadb.PersistentClient(path="/app/vector_store")
chroma_collection = db2.get_or_create_collection("LMITD2")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(vector_store)

model_name = "meta-llama/Llama-3.2-3B-Instruct"

def get_llm():
    torch.cuda.set_per_process_memory_fraction(0.1, 0)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_key)
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    tokenizer.padding_side = 'right'
    
    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    
    llm = HuggingFaceLLM(
        model_name=model_name,
        model_kwargs={
            "token": hf_key,
            "quantization_config": quantization_config
        },
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "pad_token_id": tokenizer.pad_token_id
        },
        max_new_tokens=2048,
        tokenizer_name=model_name,
        tokenizer_kwargs={"token": hf_key},
        stopping_ids=stopping_ids
    )
    
    return llm


llm = get_llm()


@app.post("/chat/")
async def api_chat_with_document(chat_query: ChatQuery):
    try:
        result = await chat_with_document(chat_query, llm)
        return result
    except Exception as e:
        print(f"Error in /chat/: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/related_documents/")
async def api_get_related_documents(thesis: ThesisTitle):
    return await get_related_documents(thesis, index)

