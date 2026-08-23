import chromadb
from chromadb import Collection
from chromadb.api import ClientAPI
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = Path(PROJECT_ROOT, 'data', 'chroma')
COLLECTION_NAME = 'magic_rules'
EMBEDDING_MODEL = 'BAAI/bge-m3'

def load_chroma() -> tuple[ClientAPI, Collection]:
    client = chromadb.PersistentClient(path = CHROMA_PATH)
    collection = client.get_collection(
        name = COLLECTION_NAME  ,
        embedding_function = None
    )

    return client, collection

def loader() -> tuple[SentenceTransformer, ClientAPI, Collection]:
    model = SentenceTransformer(EMBEDDING_MODEL, device = 'cpu')
    client, collection = load_chroma()

    return model, client, collection