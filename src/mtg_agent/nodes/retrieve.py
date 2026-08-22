from sentence_transformers import SentenceTransformer
from chromadb import Collection
from typing import TypedDict

class RetrievalResult(TypedDict):
    ids: list[str]
    documents: list[str]
    metadatas: list[dict[str, object]]
    distances: list[float]

def retrieve(
        query: str,
        model: SentenceTransformer,
        collection: Collection,
        top_k: int = 10
) -> RetrievalResult:
    query_embedding = model.encode(
        query,
        normalize_embeddings = True
    )
    query_args = {
        'query_embeddings': [query_embedding.tolist()],
        'n_results': top_k,
        'include': ['documents', 'metadatas', 'distances']
    }
    retrieval_context = collection.query(**query_args)
    return {
        'ids': (retrieval_context.get('ids') or [[]])[0],
        'documents': (retrieval_context.get('documents') or [[]])[0],
        'metadatas': (retrieval_context.get('metadatas') or [[]])[0],
        'distances': (retrieval_context.get('distances') or [[]])[0],
    }




