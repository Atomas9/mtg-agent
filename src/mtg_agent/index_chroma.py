import chromadb

from pathlib import Path
from sentence_transformers import SentenceTransformer
from mtg_agent.pdf_parser import RuleChunk


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RULE_CHUNKS_PATH = Path(PROJECT_ROOT, 'data', 'processed', 'rule_chunks.jsonl')
CHROMA_PATH = Path(PROJECT_ROOT, 'data', 'chroma')
COLLECTION_NAME = 'magic_rules'
EMBEDDING_MODEL = 'BAAI/bge-m3' 
BATCH_SIZE = 8

def load_rule_chunks(path: Path ) -> list[RuleChunk]:
    rule_chunks: list[RuleChunk] = []

    with path.open('r', encoding = 'utf-8') as file:
        for line in file:
            if not line.strip():
                continue
            rule_chunk = RuleChunk.model_validate_json(line)
            rule_chunks.append(rule_chunk)
    
    return rule_chunks

def rule_chunk_to_metadata(rule_chunk: RuleChunk) -> dict:
    return {
        "chapter_number": rule_chunk.chapter_number,
        "chapter_title": rule_chunk.chapter_title,
        "section_number": rule_chunk.section_number,
        "section_title": rule_chunk.section_title,
        "rule_number": rule_chunk.rule_number,
        "subrule_numbers": ", ".join(rule_chunk.subrule_numbers),
        "page_start": rule_chunk.page_start,
        "page_end": rule_chunk.page_end,
        "source_file": rule_chunk.source_file,
        "source_sha256": rule_chunk.source_sha256,
        "parser_version": rule_chunk.parser_version,
    }

def main():
    rule_chunks = load_rule_chunks(RULE_CHUNKS_PATH)

    model = SentenceTransformer(EMBEDDING_MODEL, device = 'cpu')

    chunk_texts = [rule_chunk.chunk_text for rule_chunk in rule_chunks]

    embeddings = model.encode(
        chunk_texts, 
        batch_size = BATCH_SIZE,
        show_progress_bar = True,
        normalize_embeddings = True,    
    )
    
    client = chromadb.PersistentClient(path = CHROMA_PATH)
    collection = client.get_or_create_collection(
        name = COLLECTION_NAME,
        embedding_function = None,
        configuration = {
            'hnsw': {
                'space': 'cosine'
            }
        }
    )

    ids = [rule_chunk.chunk_id for rule_chunk in rule_chunks]
    documents = [rule_chunk.chunk_text for rule_chunk in rule_chunks]
    metadatas = [rule_chunk_to_metadata(rule_chunk) for rule_chunk in rule_chunks]

    collection.upsert(
        ids = ids,
        embeddings = embeddings.tolist(),
        metadatas = metadatas,
        documents = documents
    )

    print(f"Chunks leídos: {len(rule_chunks)}")
    print(f"Embeddings generados: {len(embeddings)}")
    print(f"Registros en Chroma: {collection.count()}")
    print(f"Base de datos: {CHROMA_PATH.resolve()}")

if __name__ == '__main__':
    main()