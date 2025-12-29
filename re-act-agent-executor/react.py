# this file will hold our reasoning engine (logic) which we are going to use, which decides which tool to call


from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()


# triple --> function takes a float as input, and it's going to return us this float triple
@tool
def triple(num: float) -> float:
    """
    :param num: a number to triple
    :return: the number tripled ->  multiplied by 3
    """
    return 3 * float(num)


tools = [TavilySearch(max_results=1), triple]

llm = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)