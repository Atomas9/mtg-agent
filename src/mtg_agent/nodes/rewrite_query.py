from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage

OLLAMA_MODEL = 'gemma4:31b-cloud'

SYSTEM_PROMPT = '''
You rewrite the latest user request as a standalone request.

Use the previous conversation only to resolve references in the
latest user request.

Rules:
- Preserve the language of the latest user request.
- Preserve its original meaning and intent.
- Resolve pronouns and references such as "it", "that card",
  "the previous one" or equivalent expressions.
- Include only the previous context needed to understand the request.
- Preserve Magic: The Gathering card names exactly as they appear.
- You may correct obvious spelling errors only in common words.
- Never correct, translate or guess the spelling of card names.
- Copy every possible card name character for character, even if it appears
  misspelled or you know a different official spelling.
- Do not add facts that are not present in the conversation.
- Do not answer the request.
- If the conversation contains only one user request, return it without
  contextual changes, except for permitted common-word spelling corrections.
- If the latest request is already standalone, return it unchanged, except
  for permitted common-word spelling corrections.
- Return only the rewritten request as plain text.
- Do not include explanations, prefixes, quotation marks, Markdown
  or code fences.

Example:

Conversation:
User: "¿Qué hace Battlefield Raptor?"
Assistant: "Tiene flying y first strike."
User: "¿Y cómo intreractúa con Ninja of the Deep Hours?"

Rewritten request:
¿Cómo interactúa Battlefield Raptor con Ninja of the Deep Hours?

Card-name preservation example:

User request:
"¿Como funcciona first strike en Batlefield Raptor?"

Rewritten request:
¿Cómo funciona first strike en Batlefield Raptor?
'''.strip()

def rewrite_query(messages: list[BaseMessage], model_name: str = OLLAMA_MODEL) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', SYSTEM_PROMPT),
            MessagesPlaceholder('messages')
        ]
    )
    llm = ChatOllama(
        model = model_name,
        temperature = 0
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            'messages': messages
        }
    )
    return response.content.strip()
