
# in this file we're going to implement a chain that is going to determine whether the answer we get back from LLM,
# The generation is grounded in the documents




from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(temperature=0)


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# So basically the answer we will get back from the LLM langchain will format it as the Pydantic class of great hallucinations,
# which we created them, which is going to have only one attribute of binary score (bool)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""


hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

# here we are going to get an answer yes or no, If the answer is indeed grounded in the documents that we plug in.
hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader