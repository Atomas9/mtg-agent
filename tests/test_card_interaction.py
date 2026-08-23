from mtg_agent.nodes.card_interaction import build_interaction_retrieval_query


def test_interaction_query_adds_each_card_text_once() -> None:
    query = '¿Qué ocurre cuando interactúan estas cartas?'
    cards = [
        {'name': 'Card A', 'text': 'Card A rules text.'},
        {'name': 'Card A', 'text': 'Repeated edition text.'},
        {'name': 'Card B', 'text': 'Card B rules text.'},
        {'name': 'Card without text'}
    ]

    result = build_interaction_retrieval_query(query, cards)

    assert query in result
    assert result.count('Card A:') == 1
    assert 'Card A rules text.' in result
    assert 'Repeated edition text.' not in result
    assert result.count('Card B:') == 1
    assert 'Card B rules text.' in result
