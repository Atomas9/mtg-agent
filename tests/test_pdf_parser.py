from pathlib import Path

import pytest

from mtg_agent.pdf_parser import RuleChunk, validate_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULE_CHUNKS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'rule_chunks.jsonl'


def load_rule_chunks() -> list[RuleChunk]:
    with RULE_CHUNKS_PATH.open('r', encoding = 'utf-8') as file:
        return [
            RuleChunk.model_validate_json(line)
            for line in file
            if line.strip()
        ]


def test_processed_rule_chunks_are_valid() -> None:
    chunks = load_rule_chunks()

    validate_chunks(chunks)

    assert len(chunks) == 1164
    assert chunks[0].rule_number == '100.1'
    assert chunks[-1].rule_number == '905.6'


def test_validate_chunks_rejects_duplicate_rules() -> None:
    first_chunk = load_rule_chunks()[0]

    with pytest.raises(ValueError, match = 'Duplicate rules found'):
        validate_chunks([first_chunk, first_chunk])
