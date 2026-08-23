import httpx

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from typing import Literal

OLLAMA_MODEL = 'gemma4:31b-cloud'
CARDS_API_URL = 'https://api.magicthegathering.io/v1/cards'
DEFAULT_CARD_LIMIT = 5
API_TIMEOUT = 15.0
COLOR_CODES = {
    'White': 'W',
    'Blue': 'U',
    'Black': 'B',
    'Red': 'R',
    'Green': 'G'
}

SYSTEM_PROMPT = '''
You extract search filters for Magic: The Gathering cards.

Return only valid JSON using exactly this structure:

{{
  "name": null,
  "colors": [],
  "types": [],
  "subtypes": [],
  "text": null,
  "max_cmc_exclusive": null
}}

Rules:
- Write colors in English using only:
  "White", "Blue", "Black", "Red" or "Green".
- Put card types such as "Creature" in "types".
- Put creature subtypes such as "Warrior" in "subtypes".
- "Less than 2 mana" means "max_cmc_exclusive": 2.
- Use null or an empty list when a filter was not requested.
- Do not include explanations, Markdown or additional fields.
'''.strip()

USER_PROMPT = '''
User requests:

{query}
'''.strip()

class CardSearchFilters(BaseModel):
    name: str | None = None
    colors: list[
        Literal['White', 'Blue', 'Black', 'Red', 'Green']
    ] = Field(default_factory = list)
    types: list[str] = Field(default_factory=list)
    subtypes: list[str] = Field(default_factory=list)
    text: str | None = None
    max_cmc_exclusive: float | None = None

def extract_card_filters(query: str, model_name: str = OLLAMA_MODEL) -> CardSearchFilters:
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', SYSTEM_PROMPT),
            ('human', USER_PROMPT)
        ]
    )
    llm = ChatOllama(
        model = model_name,
        temperature = 0
    )
    chain = prompt | llm
    response = chain.invoke(
        {'query': query}
    )
    filters = CardSearchFilters.model_validate_json(response.content)

    return filters

def search_cards(
    filters: CardSearchFilters,
    limit: int = DEFAULT_CARD_LIMIT,
) -> list[dict]:

    params = {
        'pageSize': 100,
    }

    if filters.name:
        params['name'] = filters.name

    if filters.colors:
        params['colors'] = ','.join(
            COLOR_CODES[color]
            for color in filters.colors
        )

    if filters.types:
        params['types'] = ','.join(filters.types)

    if filters.subtypes:
        params['subtypes'] = ','.join(filters.subtypes)

    if filters.text:
        params['text'] = filters.text

    response = httpx.get(
        CARDS_API_URL,
        params = params,
        timeout = API_TIMEOUT,
    )

    response.raise_for_status()

    cards = response.json().get('cards', [])

    if filters.max_cmc_exclusive is not None:
        cards = [
            card
            for card in cards
            if card.get('cmc') is not None
            and card['cmc'] < filters.max_cmc_exclusive
        ]

    return cards[:limit]
    

