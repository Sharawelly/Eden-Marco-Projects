import sys
import os
# This forces Python to add the current folder to the list of paths it checks for modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 2. Load the .env file explicitly from the current_dir
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(current_dir, ".env"))


# Annotated --> to help us to add metadata to those type hints of TypedDict
from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
# add_messages --> to ensure that new messages are appending to the existing conversation history instead of replacing it
# so simply this function will append to a list here and updating the state
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain # those are the chains which we are going to run in each node in our langgraph graph



load_dotenv()


# data structure that every node in our graph will have access to
class MessageGraph(TypedDict):
    # after every iteration we want to keep updating these messages key after every node execution in a graph
    # se we want to keep appending to this list because every execution is going to generate a message from the AI
    # and we want to go and append and append and append it so this is the goal of state here,
    # simply a data structure to hold all of those list of messages here.
    # Annotated --> metadata that will tell langgraph how to handle state updates,
    # so once we write it like this, then tha langgraph will know that when it's going to change the state and update it
    # so instead of updating a dictionary like replacing the key, it will append new items to the value of that key here.
    # and this is because our add_messages reducer here, so the reducer is a general terminology in langgraph how do we want to update the state here,
    # and we can put here any function that we want as long as it adheres to reducer interface, and we have total flexibility of how to do it
    messages: Annotated[list[BaseMessage], add_messages]



# keys of our langgraph and those are all the node that we will need to implement in our graph
REFLECT = "reflect" # will run the reflection chain
GENERATE = "generate" # will run the generate chain






def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    # casting AI message to be HumanMessage
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)

# this function can use LLM to decide whether we want to finish or whether we want to go to another node,
# se we have an LLM to reason where do we want to go?
# but in this example we didn't use LLM, we use dummy condition
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 2:
        return END
    return REFLECT


builder.add_conditional_edges(GENERATE, should_continue)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
print(graph.get_graph().draw_mermaid())
# graph.get_graph().print_ascii()



if __name__ == '__main__':
    print("Hello langgraph")
    inputs = {
        "messages": [
            HumanMessage(
                content="""Make this tweet better:"
                                       @LangChainAI
               — newly Tool Calling feature is seriously underrated.

               After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy.

               Made a video covering their newest blog post

                                     """
            )
        ]
    }
    response = graph.invoke(inputs)
    print(response)

    print("Agent run complete.")

    # 2. NOW you can fetch the trace (Because the project now exists)
    from langsmith import Client
    import webbrowser

    client = Client()
    # It might take a split second for the server to register, but usually it's instant
    runs = list(client.list_runs(project_name="reflection-agent-tutorial", limit=1))

    if runs:
        print(f"Opening trace: {runs[0].url}")
        webbrowser.open(runs[0].url)
    else:
        print("Run finished, but trace not found yet (might need a moment to upload).")