from typing import List

# pydantic to create the object that we want to structure our output.
from pydantic import BaseModel, Field

# we want to ensure that the output we get from the LLM, then it is in a structured format.
# So we want the format to be with a response field that is having the original essay.
# We want the critique field, which is having the critique for that essay, and we want a search field,
# which will be a list of values that we should search for.
# those classes to make sure that the output format of the LLM is going to be in a object we create.
# so this file is going to hold the schemas for the output we want from LLM, and force the LLM to response output in this structure



# our critique
# the response of the LLM should match and fill up those values, so it should actually give us very concise feedback from the LLM.
class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    # superfluous --> means unnecessary information, information that doesn't add any value.
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question."""

    answer: str = Field(description="~100 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question."""

    references: List[str] = Field(
        description="Citations motivating your updated answer."
    )