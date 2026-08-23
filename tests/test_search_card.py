from mtg_agent.nodes import search_card
from mtg_agent.nodes.search_card import CardSearchFilters, search_cards


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {
            'cards': [
                {'name': 'Card A', 'cmc': 1.0},
                {'name': 'Card B', 'cmc': 1.5},
                {'name': 'Card C', 'cmc': 2.0},
                {'name': 'Card D', 'cmc': None}
            ]
        }


def test_search_cards_maps_colors_and_filters_exclusive_cmc(monkeypatch) -> None:
    captured_request = {}

    def fake_get(url: str, params: dict, timeout: float) -> FakeResponse:
        captured_request['url'] = url
        captured_request['params'] = params
        captured_request['timeout'] = timeout
        return FakeResponse()

    monkeypatch.setattr(search_card.httpx, 'get', fake_get)

    filters = CardSearchFilters(
        colors = ['White', 'Red'],
        types = ['Creature'],
        subtypes = ['Warrior'],
        max_cmc_exclusive = 2
    )

    cards = search_cards(filters)

    assert captured_request == {
        'url': search_card.CARDS_API_URL,
        'params': {
            'pageSize': 100,
            'colors': 'W,R',
            'types': 'Creature',
            'subtypes': 'Warrior'
        },
        'timeout': search_card.API_TIMEOUT
    }
    assert [card['name'] for card in cards] == ['Card A', 'Card B']
