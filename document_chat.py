from fastapi import HTTPException
from fastapi.responses import JSONResponse
from models import ChatQuery, CombinedQuery, ThesisTitle
import asyncio
import time

async def get_related_documents(thesis: ThesisTitle, index):
    try:
        related_documents = index.as_retriever(similarity_top_k=thesis.number).retrieve(thesis.title)
        
        results = []
        for doc in related_documents:
            text = doc.text
            url_start = text.find("url")
            url_end = text.find("\n", url_start)
            url = text[url_start + len("url:"):url_end].strip() if url_start != -1 else "No URL"

            abstrak_start = text.find("Abstrak:")
            abstrak_text = text[abstrak_start + len("Abstrak:"):].strip()
            kata_kata = abstrak_text.split()
            abstrak_20_kata = ' '.join(kata_kata[:20]) + '.....'

            author_start = text.find("Author:")
            judul_start = text.find("Judul:")
            judul = text[judul_start + len("Judul:"):author_start].strip()

            results.append({
                "judul": judul,
                "abstrak": abstrak_text,
                "url": url
            })

        return {"related_documents": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_academic_answer_prompt(chat_history: str, context: str, query: str, is_follow_up: bool = False) -> str:
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
    """

    if is_follow_up:
        follow_up_instruction = """
**Note:** This is a follow-up question. Make sure to reference the previous answer where relevant and clarify any new information requested by the user.
"""
        formatting_instructions += follow_up_instruction

    return f"{formatting_instructions}\n\nUser query: {prompt}\n\nFormatted response:"

async def chat_with_document(chat_query: ChatQuery, llm):
    start_time = time.time()
    try:
        limited_chat_history = chat_query.chat_history[-3:]
        chat_history = "\n".join(f"{msg.role}: {msg.content}" for msg in limited_chat_history)
        prompt = generate_academic_answer_prompt(chat_history, chat_query.context, chat_query.query)
        response = await asyncio.to_thread(llm.complete, prompt)

        process_time = time.time() - start_time
        print(f"Chat with document request took {process_time:.4f} seconds")
        
        return JSONResponse(content={"response": str(response)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
