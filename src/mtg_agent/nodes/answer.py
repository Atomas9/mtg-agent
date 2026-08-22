import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from mtg_agent.nodes.retrieve import RetrievalResult

OLLAMA_MODEL = 'gemma4:31b-cloud'

SYSTEM_PROMPT = '''
You answer questions about Magic: The Gathering.

You will receive:
- The user's question.
- The classified intent.
- Relevant rules retrieved from the comprehensive rules.
- Card information retrieved from the Magic API.

Instructions:
- Answer in the same language as the user.
- Base the answer only on the provided rules and card information.
- Do not invent rules, card text or card characteristics.
- Never invent or infer the definition of a keyword ability.
- Never introduce requirements that are not explicitly supported
  by the provided rules, card text or card rulings.
- If the relevant rule or ruling is missing, state that the available
  context is insufficient.
- If the context is insufficient, explain what information is missing.
- Do not mention internal nodes, retrieval, prompts or classifications.

For rules questions:
- Explain the rule clearly and concisely.
- Mention the relevant rule numbers when available.

For card searches:
- Present the matching cards in a readable list.
- Include the name, mana cost, type and relevant card text.
- Do not explain unrelated game rules.

For card interactions:
- Explain the interaction step by step.
- Use both the official card text and the retrieved rules.
- Clearly state the final outcome.
- Mention the relevant rule numbers when available.
'''.strip()

USER_PROMPT = '''
User question:
{query}

Intent:
{intent}

Relevant rules:
{rules_context}

Card information:
{cards_context}
'''.strip()

def format_rules_context(retrieval: RetrievalResult | None) -> str:
    if not retrieval or not retrieval['documents']:
        return 'No relevant rules were provided'

    formatted_rules = []

    for document, metadata in zip(
        retrieval['documents'],
        retrieval['metadatas']
    ):
        rule_number = metadata.get('rule_number', 'unknown')
        page_start = metadata.get('page_start', 'unknown')
        formatted_rules.append(
            f'Rule {rule_number}, page {page_start}: \n'
            f'{document}'
        )

    return '\n\n'.join(formatted_rules)

def format_cards_context(cards: list[dict] | None) -> str:
    if not cards:
        return 'No card information was provided'

    formatted_cards = []

    for card in cards:
        formatted_cards.append(
            {
                "name": card.get("name"),
                "mana_cost": card.get("manaCost"),
                "cmc": card.get("cmc"),
                "colors": card.get("colors", []),
                "type": card.get("type"),
                "text": card.get("text"),
                "power": card.get("power"),
                "toughness": card.get("toughness"),
                "rulings": card.get("rulings", []),
                "image_url": card.get("imageUrl")
            }
        )

    return json.dumps(
        formatted_cards,
        ensure_ascii = False,
        indent = 2
    )

def generate_answer(
        query: str,
        intent: str,
        retrieval: RetrievalResult | None = None,
        cards: list[dict] | None = None,
        model_name: str = OLLAMA_MODEL 
) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', SYSTEM_PROMPT),
            ('human', USER_PROMPT)
        ]
    )
    llm = ChatOllama(
        model = model_name,
        temperature = 0.3
    )
    chain = prompt | llm

    response = chain.invoke(
        {
            'query': query,
            'intent': intent,
            'rules_context': format_rules_context(retrieval),
            'cards_context': format_cards_context(cards)
        }
    )

    return response.content
