

# Generation node is going to be the last node that is going to be executed,
# We execute this node after we already retrieve the information,
# the relevant documents, after we filtered out the documents that were not relevant to our query, and even performed a search for the question that we want to answer,
# So after we have all the documents, we can augment the original query, So now it's time to simply stop everything and to send it to the LLM to answer it.
# This node is going to simply take the question and take the documents from our state and simply run the chain.

from typing import Any, Dict

from ..chains.generation import generation_chain
from ..state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    # updating the generation key in our graph state to be the generation the answer that the LLM responded.
    return {"documents": documents, "question": question, "generation": generation}