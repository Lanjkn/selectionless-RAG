# Utilitários para Banco de Dados Vetorial Sem Seleção

Este módulo fornece utilitários de alto nível para chunking semântico, extração de embeddings e busca semântica eficiente em documentos usando bancos de dados vetoriais. Ele foi projetado para suportar pipelines de RAG (retrieval-augmented generation) sem seleção manual e busca semântica refinada em aplicações de PLN.

## Funcionalidades

- **Chunking Semântico**: Usa [`chonkie.SemanticChunker`](https://github.com/minishlab/chonkie) para dividir grandes textos em chunks semanticamente significativos, melhorando a precisão da recuperação.
- **Extração de Embeddings**: Utiliza [`utility_pack.embeddings.extract_embeddings`] para gerar representações vetoriais tanto para os chunks quanto para documentos inteiros.
- **Armazenamento Vetorial**: Armazena e gerencia embeddings usando [`utility_pack.vector_storage.MiniVectorDB`], permitindo busca por similaridade rápida e armazenamento persistente.
- **Busca Sem Seleção**: Permite buscar documentos relevantes sem seleção manual, retornando os mais similares semanticamente a uma consulta.
- **Recuperação Contextual**: Suporta extração de contexto a partir de conjuntos filtrados de documentos, útil para tarefas como perguntas e respostas ou sumarização.

## Funções Principais

- `realizar_chunking_semantico_de_texto(texto, chunk_size=512)`: Divide o texto em chunks semânticos.
- `retirar_embeddings_e_salvar_no_minivector_db(texto, metadados, doc_name)`: Extrai embeddings dos chunks e do documento inteiro, salvando-os nos bancos vetoriais.
- `buscar_semanticamente_entre_documentos(query, k=5)`: Encontra os k documentos mais relevantes para uma consulta usando similaridade semântica.
- `realizar_retirada_de_contexto_de_documentos_apropriados(docs_para_filtrar, query, k=5)`: Recupera os contextos mais relevantes de um conjunto filtrado de documentos.

## Benefícios

- **Busca Semântica Automatizada**: Não é necessário etiquetar ou selecionar manualmente—basta consultar e recuperar o conteúdo mais relevante.
- **Armazenamento Eficiente**: Embeddings são armazenados de forma persistente, permitindo buscas rápidas e escaláveis em grandes coleções de documentos.
- **Recuperação Contextual**: Retorna não apenas os IDs dos documentos, mas também o texto original e metadados para processamento posterior mais rico.
- **Integração Fácil**: Projetado para funcionar com outros módulos do workspace, como `utility_pack` e `chonkie`.

## Exemplo de Uso

```python
from vector_db import retirar_embeddings_e_salvar_no_minivector_db, buscar_semanticamente_entre_documentos

# Indexar um documento
doc_id = retirar_embeddings_e_salvar_no_minivector_db("Algum texto...", {"autor": "Alice"}, "doc1")

# Buscar documentos relevantes
query = "O que é chunking semântico?"
k = 3

documentos_apropriados = buscar_semanticamente_entre_documentos(query, k)
contextos = realizar_retirada_de_contexto_de_documentos_apropriados(
    docs_para_filtrar=documentos_apropriados,
    query=query,
    k=k
)
print(contextos)
```

## Quando Usar

- Construção de pipelines RAG
- Busca semântica de documentos
- Recuperação contextual para chatbots ou sistemas de perguntas e respostas
- Qualquer tarefa de PLN que exija recuperação rápida, precisa e sem seleção manual de grandes corpora de texto

---
*Veja [`vector_db.py`](vector_db.py) para detalhes da implementação.*