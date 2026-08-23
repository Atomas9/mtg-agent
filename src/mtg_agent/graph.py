from typing import TypedDict, Literal, Annotated
from chromadb import Collection
from sentence_transformers import SentenceTransformer

from mtg_agent.nodes.prepare_turn import prepare_turn
from mtg_agent.nodes.rewrite_query import rewrite_query
from mtg_agent.nodes.classifier import classifier
from mtg_agent.nodes.retrieve import RetrievalResult, retrieve
from mtg_agent.nodes.search_card import (
    CardSearchFilters, 
    extract_card_filters, 
    search_cards
)
from mtg_agent.nodes.card_interaction import (
    CardInteractionEntities, 
    extract_card_names, 
    get_cards_by_names, 
    build_interaction_retrieval_query
)
from mtg_agent.nodes.answer import generate_answer

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from functools import partial

class GraphState(TypedDict, total = False):
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str
    query: str
    intent: Literal[
        'rules',
        'card_search',
        'card_interaction',
        'custom_card',
        'out_of_scope'
    ]
    card_filters: dict[str, object]
    card_names: dict[str, object]
    cards: list[dict]
    retrieval: RetrievalResult | None
    answer: str

# ----------
# NODES
# ----------
def prepare_turn_node(state: GraphState) -> dict:
    response = prepare_turn(state['messages'])
    return response

def rewrite_query_node(state: GraphState) -> dict:
    response = rewrite_query(state['messages'])
    return {'query': response}

def classifier_node(state: GraphState) -> dict:
    response = classifier(state['query'])
    return {'intent': response.intent}

def retrieve_rules_node(
        state: GraphState,
        emb_model: SentenceTransformer,
        collection: Collection,
        top_k: int = 10
) -> dict:
    response = retrieve(state['query'], emb_model, collection, top_k)
    return {'retrieval': response}

def extract_card_filters_node(state: GraphState) -> dict:
    response = extract_card_filters(state['query'])
    return {'card_filters': response.model_dump(mode = 'json')}

def search_cards_node(state: GraphState) -> dict:
    filters = CardSearchFilters.model_validate(state['card_filters'])
    response = search_cards(filters)
    return {'cards': response}

def extract_card_names_node(state: GraphState) -> dict:
    response =extract_card_names(state['query'])
    return {'card_names': response.model_dump(mode = 'json')}

def get_cards_by_names_node(state: GraphState) -> dict:
    names = CardInteractionEntities.model_validate(state['card_names'])
    response = get_cards_by_names(names)
    return {'cards': response}

def retrieve_interaction_node(
        state: GraphState,
        emb_model: SentenceTransformer,
        collection: Collection,
        top_k: int = 10
) -> dict:
    new_query = build_interaction_retrieval_query(state['query'], state['cards'])
    response = retrieve(new_query, emb_model, collection, top_k)
    return {'retrieval': response}

def generate_answer_node(state: GraphState):
    response = generate_answer(
        query = state['query'],
        intent = state['intent'],
        retrieval = state.get('retrieval'),
        cards = state.get('cards')
    )
    return {
        'answer': response,
        'messages': [
            AIMessage(content = response)
        ]
    }

# def custom_card_node(state: GraphState)

def out_of_scope_node(state: GraphState) -> dict:
    response = (
        'Lo siento, solo puedo responder preguntas sobre '
        'reglas y cartas de Magic: The Gathering.'
    )
    return {
        'answer': response,
        'messages': [
            AIMessage(content = response)
        ]
    }

# ----------
# ROUTING LOGIC
# ----------
def route_after_classifier(
        state: GraphState
) -> Literal['rules', 'card_search', 'card_interaction', 'custom_card', 'out_of_scope']:
    return state['intent']

# ---------
# GRAPH
# ---------
def create_graph(checkpointer, emb_model: SentenceTransformer, collection: Collection):
    top_k = 10
    retrieve_rules_node_conf = partial(
        retrieve_rules_node,
        emb_model = emb_model,
        collection = collection,
        top_k = top_k
    )
    retrieve_interaction_node_conf = partial(
        retrieve_interaction_node,
        emb_model = emb_model,
        collection = collection,
        top_k = top_k
    )

    graph = StateGraph(GraphState)

    graph.add_node('PrepareTurn', prepare_turn_node)
    graph.add_node('RewriteQuery', rewrite_query_node)
    graph.add_node('Classifier', classifier_node)
    graph.add_node('RetrieveRules', retrieve_rules_node_conf)
    graph.add_node('ExtractCardFilters', extract_card_filters_node)
    graph.add_node('SearchCards', search_cards_node)
    graph.add_node('ExtractCardNames', extract_card_names_node)
    graph.add_node('GetCardsByName', get_cards_by_names_node)
    graph.add_node('RetrieveInteraction', retrieve_interaction_node_conf)
    graph.add_node('GenerateAnswer', generate_answer_node)
    #graph.add_node('CustomCard', custom_card_node)
    graph.add_node('OutOfScope', out_of_scope_node)

    graph.add_edge(START, 'PrepareTurn')
    graph.add_edge('PrepareTurn', 'RewriteQuery')
    graph.add_edge('RewriteQuery', 'Classifier')
    graph.add_conditional_edges(
        'Classifier',
        route_after_classifier,
        {
            'rules': 'RetrieveRules',
            'card_search': 'ExtractCardFilters',
            'card_interaction': 'ExtractCardNames',
            'custom_card': 'OutOfScope',
            'out_of_scope': 'OutOfScope'
        }
    )
    graph.add_edge('RetrieveRules', 'GenerateAnswer')
    graph.add_edge('ExtractCardFilters', 'SearchCards')
    graph.add_edge('SearchCards', 'GenerateAnswer')
    graph.add_edge('ExtractCardNames', 'GetCardsByName')
    graph.add_edge('GetCardsByName', 'RetrieveInteraction')
    graph.add_edge('RetrieveInteraction', 'GenerateAnswer')
    graph.add_edge('GenerateAnswer', END)
    graph.add_edge('OutOfScope', END)
    #graph.add_edge('CustomCard', )

    return graph.compile(checkpointer = checkpointer)