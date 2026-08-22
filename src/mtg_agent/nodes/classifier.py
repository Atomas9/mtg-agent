from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel
from typing import Literal

OLLAMA_MODEL = 'gemma4:31b-cloud'
SYSTEM_PROMPT = '''
You classify requests for a Magic: The Gathering assistant.

Available intents:

- rules:
  General questions about game rules, phases, mana, combat,
  timing, zones or keyword abilities that do not require looking
  up a specific card.

- card_search:
  Requests to find, recommend, display or obtain information
  about one or more cards.

- card_interaction:
  Questions asking what happens in a game situation involving
  one or more specific cards or effects.

- custom_card:
  Requests to invent or create a new custom card.

- out_of_scope:
  Requests unrelated to Magic: The Gathering.

Important distinctions:
- "How does first strike work?" is rules.
- "Show me Battlefield Raptor" is card_search.
- "How does first strike work with Battlefield Raptor in combat?"
  is card_interaction.
- "Create a white-red Han Solo card" is custom_card.

Classify the user request.

Return only valid JSON using exactly this structure:

{{
  "intent": "<intent>"
}}

The value of "intent" must be exactly one of:
- "rules"
- "card_search"
- "card_interaction"
- "custom_card"
- "out_of_scope"

Do not answer the question. Do not include explanations, Markdown,
code fences or additional fields.
'''.strip()

USER_PROMPT = '''
User request:

{query}
'''.strip()

class ClassifierDecision(BaseModel):
    intent: Literal[
        'rules',
        'card_search',
        'card_interaction',
        'custom_card',
        'out_of_scope'
    ]

def classifier(query: str, model_name: str = OLLAMA_MODEL) -> ClassifierDecision:
    if not query.strip():
        raise ValueError('La pregunta no puede estar vacía')
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', SYSTEM_PROMPT),
            ('human', USER_PROMPT)
        ]
    )
    llm = ChatOllama(
        model = model_name,
        temperature = 0,
    )
    chain = prompt | llm
    response = chain.invoke(
        {'query': query}
    )
    decision = ClassifierDecision.model_validate_json(response.content)

    return decision

