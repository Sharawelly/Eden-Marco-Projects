# this node is going to extract the question that the user asked, and it's going to retrieve the relevant documents for that state,
# So that is going to be using our vector store semantic search capabilities
# And after this node we should update the state documents to hold the relevant documents from our vector store.

from typing import Any, Dict

from ..state import GraphState
from ...ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    # will do the semantic search and get us all the relevant documents.
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

