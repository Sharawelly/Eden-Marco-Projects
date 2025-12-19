# this node is going to take as an input the I message, which is going to have the information of the wanted search query,
# And it’s going to run tavily to get us real time results and real time information from the web.


from dotenv import load_dotenv



load_dotenv()

# StructuredTool --> allow us to convert a Python function into a tool that can be used by LLMs,
# So it’s going to take that function and provide to the LLM a structured schema for the function, which will help the LLM understand how to use this tool.
from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
# ToolNode --> it’s going to look in the state for the messages key,
# It’s going to check the last message, and it’s then going to see if there’s any tool calls that were decided by LLM,
# and if there are, it's going to execute those tools for us, and it can do this even in parallel, so this save us tons of work.
from langgraph.prebuilt import ToolNode

from schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)


# **kwargs --> in case the LLM is going to fail some other values (some queries), so we don't get an error in that case
def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries."""
    return tavily_tool.batch([{"query": query} for query in search_queries])


# here the tool node is going to run two different tools:
# one tool is going to be the search tool that originated from the first creation of the research,
# And the other tool is going to be a search query that originated from the reflection revision.
# this tool is going to examine the state, it's going to check the last message, and if there is a tool call it's going to execute the relevant tool call.
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__),
    ]
)