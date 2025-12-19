
# hub --> download the prompt from it.
from langchain_classic import hub

# StrOutputParser --> take our message and it's going to get the content from it and turn it into a string.
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
prompt = hub.pull("rlm/rag-prompt")

# We're going to create a generation chain where we pipe the prompt into the LM,
# and then we pipe the results into StrOutput,
# So once we invoke this chain with the documents and the question, we're supposed to get the answer.
generation_chain = prompt | llm | StrOutputParser()