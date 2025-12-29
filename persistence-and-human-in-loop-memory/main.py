from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
# MemorySaver --> checkpoint which stores the state after each node’s execution,
# However, it stores it in memory and this storage type is ephemeral, so it will be gone upon each run of our graph's execution,
# However, it's a good starting point to start and having a feeling of those objects that are checkpointed.
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

class State(TypedDict):
    input: str
    user_feedback: str


def step_1(state: State) -> None:
    print("---Step 1---")


def human_feedback(state: State) -> None:
    print("---human_feedback---")


def step_3(state: State) -> None:
    print("---Step 3--")


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("human_feedback", human_feedback)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "human_feedback")
builder.add_edge("human_feedback", "step_3")
builder.add_edge("step_3", END)


memory = SqliteSaver.from_conn_string("checkpoints.sqlite")


# because of {interrupt_before=["human_feedback"]} when we execute the graph, before we execute the human feedback node,
# we'll stop the graph execution, And because we checkpointed the state of the graph and at what point we stopped,
# then we can go and get an input from the user (get some human feedback),
# And then we can go and actually resume our graph execution from exactly where we stopped,
# So this is all thanks to the check pointer which is helping us to remember where we stopped and what was the state,
# And the use case here is when we have user facing applications, and we want to integrate human feedback from our agent, then this is a very useful technique.
graph = builder.compile(checkpointer=memory, interrupt_before=["human_feedback"])

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    # thread_id --> session id or conversation id
    thread = {"configurable": {"thread_id": "777"}}

    initial_input = {"input": "hello world"}

    for event in graph.stream(initial_input, thread, stream_mode="values"):
        print(event)

    print(graph.get_state(thread).next)

    user_input = input("Tell me how you want to update the state: ")

    graph.update_state(thread, {"user_feedback": user_input}, as_node="human_feedback")

    print("--State after update--")
    print(graph.get_state(thread))

    print(graph.get_state(thread).next)

    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)