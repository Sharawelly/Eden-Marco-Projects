from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    # we have a question in the state because we always want to reference it whether to determine if the documents retrieved are relevant to the question, or even to what to search online.
    question: str
    # generated answer
    generation: str
    # tell us whether we need to search online for extra results or not.
    web_search: bool
    # We want to save the documents that are going to help us answer this question,
    # So those are going to be the retrieved documents or the documents that we get back from the search result.
    documents: List[str]