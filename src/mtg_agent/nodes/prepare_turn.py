from langchain_core.messages import BaseMessage, HumanMessage

def prepare_turn(messages: list[BaseMessage]) -> dict[str, object]:
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue

        return {
            'user_query': message.content,
            'query': message.content,
            'card_filters': {},
            'card_names': {},
            'cards': [],
            'retrieval': None,
            'answer': ''
        }