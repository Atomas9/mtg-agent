from langchain_core.messages import AIMessage, HumanMessage

from mtg_agent.nodes.prepare_turn import prepare_turn


def test_prepare_turn_uses_latest_user_message_and_resets_turn_data() -> None:
    messages = [
        HumanMessage(content = '¿Qué hace first strike?'),
        AIMessage(content = 'Respuesta anterior'),
        HumanMessage(content = '¿Y cómo funciona con esta carta?')
    ]

    result = prepare_turn(messages)

    assert result == {
        'user_query': '¿Y cómo funciona con esta carta?',
        'query': '¿Y cómo funciona con esta carta?',
        'card_filters': {},
        'card_names': {},
        'cards': [],
        'retrieval': None,
        'answer': ''
    }
