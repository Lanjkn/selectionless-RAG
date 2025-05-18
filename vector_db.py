from utility_pack.vector_storage import MiniVectorDB
from utility_pack.embeddings import extract_embeddings, EmbeddingType
from utility_pack.text import get_uuid, remove_stopwords, compress_text
import numpy as np

# Initialize a MiniVectorDB instance
vector_db_for_selectionless_searching = MiniVectorDB(storage_file='my_vector_db_selection_less.pkl')
vector_db = MiniVectorDB(storage_file='my_vector_db.pkl')

from chonkie import SemanticChunker

# Basic initialization with default parameters
chunker = SemanticChunker(
    embedding_model="minishlab/potion-base-8M",  # Default model
    threshold=0.5,                               # Similarity threshold (0-1) or (1-100) or "auto"
    chunk_size=512,                              # Maximum tokens per chunk
    min_sentences=1                              # Initial sentences per chunk
)

def realizar_chunking_semantico_de_texto(texto, chunk_size=512):
    chunks = chunker.chunk(
        text=texto,
    )
    return chunks

def retirar_embeddings_e_salvar_no_minivector_db(texto, metadados, doc_name):
    """
    Extrai embeddings do texto e salva no MiniVectorDB.
    
    Args:
        texto (str): Texto para o qual as embeddings serão extraídas.
        metadados (dict): Metadados associados ao texto.
        doc_name (str): Nome do documento para identificação.
    """
    unique_id = doc_name + get_uuid()

    metadados['id'] = doc_name 
    metadados['documento'] = doc_name
    metadados['texto_original'] = texto

    texto_comprimido = compress_text(texto)
    texto_processado_e_enxuto = remove_stopwords(texto_comprimido)


    chunks_semanticos = realizar_chunking_semantico_de_texto(texto)
    chunks_semanticos = [chunk.text for chunk in chunks_semanticos]
    embeddings = extract_embeddings(chunks_semanticos, embedding_type=EmbeddingType.SEMANTIC)

    for i, chunk in enumerate(chunks_semanticos):
        vector_db.store_embedding(
            unique_id=unique_id + f"_{i}",
            embedding=embeddings[i],
            metadata_dict=metadados
        )

    # Extrair embeddings
    embeddings_resumo = extract_embeddings([texto_processado_e_enxuto], embedding_type=EmbeddingType.SEMANTIC)[0]

    vector_db_for_selectionless_searching.store_embedding(
        unique_id=unique_id,
        embedding=embeddings_resumo,
        metadata_dict=metadados
    )

    print(f"Embeddings for {doc_name} saved with ID: {unique_id}")

    vector_db_for_selectionless_searching.persist_to_disk()
    vector_db.persist_to_disk()

    return unique_id

def buscar_semanticamente_entre_documentos(query, k = 5):
    """
    Busca semanticamente entre documentos no MiniVectorDB.
    
    Args:
        query (str): Consulta para busca.
        k (int): Número de resultados a serem retornados.
    
    Returns:
        list: Lista de IDs dos documentos mais relevantes.
    """

    # Extrair embeddings da consulta
    query_embedding = extract_embeddings([query], embedding_type=EmbeddingType.SEMANTIC)
    
    # Buscar no MiniVectorDB
    results = vector_db_for_selectionless_searching.find_most_similar(query_embedding[0], k=k)

    ids_para_retornar = []
    ids, distances, metadatas = results

    for ids, distances, metadatas in zip(ids, distances, metadatas):
        ids_para_retornar.append(metadatas['documento'])


    return ids_para_retornar

def realizar_retirada_de_contexto_de_documentos_apropriados(docs_para_filtrar, query, k = 5):
    """
    Realiza a retirada de contexto de documentos apropriados.
    
    Args:
        ids (list): Lista de IDs dos documentos.
        query (str): Consulta para busca.
        k (int): Número de resultados a serem retornados.
    
    Returns:
        list: Lista de IDs dos documentos mais relevantes.
    """

    # Extrair embeddings da consulta
    query_embedding = extract_embeddings([query], EmbeddingType.SEMANTIC)[0]

    # Buscar no MiniVectorDB
    or_filters = [
        {
            "documento": documento,
        }
        for documento in docs_para_filtrar
    ]

    results = vector_db.find_most_similar(query_embedding, or_filters=or_filters, k=k)

    ids, distances, metadatas = results

    resultados_reais = []
    for ids, distances, metadatas in zip(ids, distances, metadatas):
        resultados_reais.append({
            "distancia": float(distances),
            "texto": metadatas['texto_original'],
            "documento_nome": metadatas['documento'],
        })

    return resultados_reais

