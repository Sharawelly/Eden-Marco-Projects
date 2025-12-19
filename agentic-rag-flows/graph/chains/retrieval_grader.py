# So after we've implemented the retrieve node, we're going to implement now the document grader node,
# So when we enter this node we have in our state the retrieve documents,
# So now we want to iterate over those documents and to determine whether they are indeed relevant to our question or not,
# So for that we're going to be writing a retrieval grader chain, which is going to use structured output from our LLM,
# and turning it into a Pydantic object that will have the information whether this document is relevant or not,
# And if the document is not relevant, we want to filter it out and keep only the documents which are relevant to question,
# and if not all documents are relevant, So this means that at least one document is not relevant to our query,
# then we want to mark the web search flag in tha GraphState tobe true, So we will go in later search for this term.


# And this chain is going to receive as an input the original question and the retrieve document, And it's going to determine whether the document is relevant to the questions or not,
# And we're going to be running this chain for each document we retrieve.




from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

# create chain
retrieval_grader = grade_prompt | structured_llm_grader