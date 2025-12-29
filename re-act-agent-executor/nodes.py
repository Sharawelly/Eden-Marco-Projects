from dotenv import load_dotenv

# MessagesState --> simple object which has the dictionary key messages and that messages are going to be a list of (BaseMessage)
# so this is going to keep track of the conversation history and all the messages that we have in that conversation history. (like human messages, ai messages, tool messages, etc.)
from langgraph.graph import MessagesState

# ToolNode --> it's going to execute tools, So it's going to check the last message between the agent and human,
# And if that last message is an AI message that has a valid tool call, it's going to go and execute that tool call, assuming that tools is initiated with the tool node object,
# So if the agent decides to execute a search or decides to run the triple function, then those are going to run inside this tool node.
from langgraph.prebuilt import ToolNode

from react import llm, tools

load_dotenv()

SYSTEM_MESSAGE = """
You are a helpful assistant that can use tools to answer questions.
"""


# MessagesState --> is dictionary that has the key of messages
def run_agent_reasoning_engine(state: MessagesState) -> MessagesState:
    response = llm.invoke(
        [{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]]
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)