from ollama import generate
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from uuid import uuid4

import base64

IMAGE_MODEL = 'x/z-image-turbo'

PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGES_PATH = PROJECT_ROOT / 'images'

PROMPT = '''
Create an original fantasy trading card illustration.

The result must:
- Have a vertical portrait composition.
- Represent the character or scene described by the user.
- Use a detailed fantasy illustration style.
- Be suitable as artwork for a trading card.
- Not include official Magic: The Gathering logos.
- Not copy an existing card or character.
- Not include text, borders or a card frame.

User description:
{query}
'''.strip()

def generate_custom_card(query: str, model_name: str = IMAGE_MODEL) -> Path:
    prompt = PromptTemplate.from_template(PROMPT)
    formatted_prompt = prompt.format(query = query)
    response = generate(
        model = model_name,
        prompt = formatted_prompt,
        width = 768,
        height = 1024
    )

    if not response.image:
        raise RuntimeError('El modelo no ha devuelto ninguna imagen')
    
    IMAGES_PATH.mkdir(parents = True, exist_ok = True)

    image_path = (
        IMAGES_PATH
        / f'custom_card_{uuid4().hex}.png'
    )

    image_path.write_bytes(
        base64.b64decode(response.image)
    )

    return image_path

