
import os
from fastapi import FastAPI, UploadFile, File

from document_extraction import TextExtractor
from vector_db import buscar_semanticamente_entre_documentos, realizar_retirada_de_contexto_de_documentos_apropriados, retirar_embeddings_e_salvar_no_minivector_db

from openai import OpenAI
OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

@app.get("/")
def read_root():
    return {"msg": "it works!"}

@app.post("/upload_document")
def upload_document_endpoint(file: UploadFile = File(...)):
    """
    Endpoint to upload a document.
    """
    # Call the function to handle the uploaded file
    file_path = file.filename
    file_path = "./uploads/" + file_path
    file_content = file.file.read()
    if not os.path.exists("./uploads"):
        os.makedirs("./uploads")

    with open(file_path, "wb") as f:
        f.write(file_content)

    text_extractor = TextExtractor(file_path)
    text = text_extractor.extract_text()

    retirar_embeddings_e_salvar_no_minivector_db(
        texto=text,
        metadados={"file_name": file_path},
        doc_name=file_path
    )

@app.get("/selectionless_context_search")
def selectionless_search_endpoint(query: str, k: int = 5):
    """
    Endpoint to perform a selectionless search.
    """
    # Call the function to perform the search
    documentos_apropriados = buscar_semanticamente_entre_documentos(query, k)

    contexto = realizar_retirada_de_contexto_de_documentos_apropriados(
        docs_para_filtrar=documentos_apropriados,
        query=query,
        k=k
    )
    
    return {"context": contexto}

@app.get("/selectionless_chat")
def selectionless_chat_endpoint(query: str, k: int = 1):
    """
    Endpoint to perform a selectionless chat.
    """
    # Call the function to perform the chat
    documentos_apropriados = buscar_semanticamente_entre_documentos(query, k)

    contextos = realizar_retirada_de_contexto_de_documentos_apropriados(
        docs_para_filtrar=documentos_apropriados,
        query=query,
        k=k
    )

    contexto_para_llm = ""

    for contexto in contextos:
        contexto_para_llm += contexto['texto'] + "\n\n"

    prompt = f"""
    Given the following context:
    <|context|>
    {contexto_para_llm}
    <|endofcontext|>
    \n
    \n
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    resposta = response.choices[0].message.content

    return {
        "resposta": resposta,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
