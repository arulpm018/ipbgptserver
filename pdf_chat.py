import os
import time
from fastapi import UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from llama_index.core import Document, VectorStoreIndex
from models import ThesisTitle, ChatQuery, Query, CombinedQuery

class PDFManager:
    def __init__(self):
        self.pdf_index = None
        self.pdf_text = ""

    def reset(self):
        self.pdf_index = None
        self.pdf_text = ""

    def generate_academic_answer_prompt(self, chat_history, context, query):
        prompt = f"""
You are a knowledgeable and friendly assistant that answers questions based on academic paper abstracts from a university repository.
Provide a thoughtful, detailed, and clear answer that explains the information from the abstract in a way that is easy to understand. 
Make sure to elaborate on key points and provide examples or context where appropriate, even if the abstract itself is brief. Avoid overly short responses and strive to give the user a comprehensive answer. 
If the information provided in the abstract fully addresses the question, end the response with this token: '<|reserved_special_token_0|>'

Chat history:\n{chat_history}\n\n
Context: {context}\n\n
Question: {query}\n\n
Answer:
        """

        formatting_instructions = """
# Formatting Instructions
Format ALL responses consistently using these guidelines:
1. Use ONLY Markdown syntax for ALL formatting.
2. NEVER USE ``` for text, JUST USE THAT FOR CODE
3. NEVER use HTML or CSS for styling.
4. Structure your response as follows:
   - Brief summary/introduction (1-4 sentences)
   - Use #### for main sections, ##### for subsections
5. For lists:
   - Use 1., 2., 3. for sequential or prioritized items
   - Use - for unordered lists
   - Consistent indentation for nested lists
6. Use **bold** for important terms, *italics* sparingly
7. For code:
   - Use inline code for short snippets
   - Use code blocks with language specification for longer code segments
8. For quotes, use > at the beginning of each line.
9. Always add a blank line between paragraphs and list items.
        """
        
        final_prompt = f"{formatting_instructions}\n\nUser query: {prompt}\n\nFormatted response:"
        return final_prompt

    async def upload_pdf(self, file: UploadFile = File(...)):
        start_time = time.time()
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF file.")
        
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        
        try:
            reader = PdfReader(file.filename)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            document = Document(text=text)
            self.pdf_index = VectorStoreIndex.from_documents([document])
            process_time = time.time() - start_time
            print(f"Upload PDF and convert to vectorstore request took {process_time:.4f} seconds")
            return JSONResponse(content={"message": "PDF uploaded and indexed successfully"})
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        
        finally:
            os.remove(file.filename)

    async def chat_with_pdf(self, combined_query: CombinedQuery, llm):
        if self.pdf_index is None:
            raise HTTPException(status_code=400, detail="No PDF has been uploaded and indexed yet.")
        
        try:
            start_time = time.time()
            retriever = self.pdf_index.as_retriever(similarity_top_k=4)
            retrieved_nodes = retriever.retrieve(combined_query.query)
            context = "\n".join([node.text for node in retrieved_nodes])
            
            chat_history = "\n".join([f"{msg.role}: {msg.content}" for msg in combined_query.chat_history])
            prompt = self.generate_academic_answer_prompt(chat_history, context, combined_query.query)
  
            response = llm.complete(prompt)
            process_time = time.time() - start_time
            print(f"Chat with PDF request took {process_time:.4f} seconds")
            return JSONResponse(content={"response": str(response)})

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Initialize a single instance of PDFManager to manage state
pdf_manager = PDFManager()
