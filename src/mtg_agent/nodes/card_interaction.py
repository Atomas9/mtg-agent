import httpx

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field

OLLAMA_MODEL = 'gemma4:31b-cloud'
CARDS_API_URL = 'https://api.magicthegathering.io/v1/cards'
API_TIMEOUT = 15.0

SYSTEM_PROMPT = '''
You extract the names of Magic: The Gathering cards explicitly
mentioned in a card interaction question.

Return only valid JSON using exactly this structure:

{{
  "names": [
    "<card name>"
  ]
}}

Rules:
- Extract only card names explicitly mentioned by the user.
- Keep the complete card name.
- Preserve the card names as written by the user.
- Do not include game mechanics, abilities, player names or other terms.
- Do not invent card names.
- Do not include the same card more than once.
- If no card name is explicitly mentioned, return an empty list.
- Do not answer the question.
- Do not include explanations, Markdown, code fences or additional fields.

Example:

User request:
"If Battlefield Raptor deals first-strike damage, can I use
Ninja of the Deep Hours before normal combat damage?"

Output:

{{
  "names": [
    "Battlefield Raptor",
    "Ninja of the Deep Hours"
  ]
}}
'''.strip()

USER_PROMPT = '''
User request:

{query}
'''.strip()

class CardInteractionEntities(BaseModel):
    names: list[str] = Field(default_factory = list)

def extract_card_names(
        query: str,
        model_name: str = OLLAMA_MODEL
) -> CardInteractionEntities:
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
    names = CardInteractionEntities.model_validate_json(response.content)
    return names

def get_cards_by_names(
        card_names: CardInteractionEntities
) -> list[dict]:
    if not card_names.names:
        return []

    names_list = '|'.join(card_names.names)
    params = {
        'name': names_list,
        'pageSize': 100
    }

    response = httpx.get(
        CARDS_API_URL,
        params = params,
        timeout = API_TIMEOUT
    )

    response.raise_for_status()

    cards = response.json().get('cards', [])
    
    return cards