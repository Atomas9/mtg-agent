from mtg_agent.graph import create_graph
from pathlib import Path
from uuid import uuid4
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage
from mtg_agent.core import loader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / 'data'
    / 'checkpoints'
    / 'langgraph.sqlite'
)

def main() -> None:
    CHECKPOINT_PATH.parent.mkdir(
        parents = True,
        exist_ok = True
    )

    conversation_id = str(uuid4())

    config = {
        'configurable': {
            'thread_id': conversation_id
        }
    }

    emb_model, chroma_client, collection = loader()

    try:
        with SqliteSaver.from_conn_string(
            str(CHECKPOINT_PATH)
        ) as checkpointer:
            graph = create_graph(
                checkpointer = checkpointer,
                emb_model = emb_model,
                collection = collection
            )

            print(f'ID de conversación: {conversation_id}')

            while True:
                query = input('\nEscribe tu pregunta ("salir" para terminar):').strip()
                if query.lower() in {'salir', 'exit', 'quit'}:
                    break

                if not query:
                    print('La pregunta no puede estar vacía')
                    continue

                state = graph.invoke(
                    {
                        'messages': [
                            HumanMessage(content = query)
                        ]
                    },
                    config = config
                )

                print(f"\n{state['answer']}")

    finally:
        chroma_client.close()

if __name__ == '__main__':
    main()

                