from typing import List

# those are the classes that are going to populate our state objects in our langgraph graph.
from langchain_core.messages import BaseMessage, ToolMessage
# MessageGraph --> built in graph with the state equals to a bunch of list of messages (based messages or tool messages) anything that will inherit from base message.
from langgraph.graph import END, MessageGraph

# we want from our chains file to import the Revisor chain and the first responder chain,
# So those are linked chain chains which will run under our graph nodes.
from chains import revisor, first_responder

# this will be the node which will do the tavily searching for real time data.
from tool_executor import execute_tools


# this means we’re going to only make two iterations in the critique and revision node.
MAX_ITERATIONS = 2


builder = MessageGraph()

# we will add a new node and the key is going to be "draft",
# And the executable is going to be the first responder chain which is going to generate the first draft
# alongside with a built-in critique and some search tools that we want to search for.
builder.add_node("draft", first_responder)

# execute_tools --> takes the input of the state And we’ll run all the tavily search queries and give us the result of some real time data from the web.
builder.add_node("execute_tools", execute_tools)

builder.add_node("revise", revisor)
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


# this function is going to receive a state and return us the string,
# And this function is going to be run after the Reviser node, and it's going to decide which node are we going to next
# are we going to finish and to output the user the answer? or we're going to continue with another iteration of tool execution and then revision.
def event_loop(state: List[BaseMessage]) -> str:
    # count the number of tools visits we had (how many times we run the searches),
    # And because we implemented the execute tools function to return us an object of tool message,
    # then we're simply going to count the number of tool messages objects we have in our state.
    # And this would represent how many times did we iterate it so far.
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop, {END:END, "execute_tools":"execute_tools"})
builder.set_entry_point("draft")
graph = builder.compile()

print(graph.get_graph().draw_mermaid())


res = graph.invoke(
    "Write about AI-Powered SOC / autonomous soc  problem domain, list startups that do that and raised capital."
)
print(res[-1].tool_calls[0]["args"]["answer"])
print(res)