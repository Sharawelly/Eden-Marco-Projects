import datetime
from dotenv import load_dotenv

load_dotenv()

# handle output from the function calling from openai
from langchain_core.output_parsers.openai_tools import (
    # both output parsers are going to take back the response,
    # we get the LLM with the function calling invocation,
    # and it's going to take the function calling invocation, and it's going to either transform it into a JSON to a dictionary or a pydantic object
    JsonOutputToolsParser,
    PydanticToolsParser
)

# ChatPromptTemplate --> hold all our history of our agent iteration,
# so we are going to append messages through that
# MessagesPlaceholder --> placeholder for new messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion, ReviseAnswer




llm = ChatOpenAI(model="o4-mini")
# return us the function call we got back from LLM and transform it into dictionary.
parser = JsonOutputToolsParser(return_id=True)
# this will take the response from the LLM, It’s going to search for the function calling invocation and it’s going to parse it and transform it into an answer question object,
# so it's going to take the answer from LLM And it’s going to create an answer question object that we can easily work with.
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        # 1. {first_instruction} --> this first part is a placeholder of the first instruction, And here .we’re going to plug in to simply write a 250 word essay.
        # 2. this critique is going to be used later by the reviser agent.
        # 3. this is only the search query, so this is not yet the search result,
        # And we’re going to use the search queries in the tool execution node where we’ll be leveraging Tavileh search engine.
        (
            "system",
            """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
        ),
        # prompt technique to reuse this prompt template
        # this prompt template is also going to be used by our Reviser node,
        # the node which is going to take now all the information and it's going to rewrite the article,
        # So because we’re going to be using that in our Revisor agent, which is going to be keep revising and critiquing,
        # revising and critiquing, then we also want to pass in the message placeholder of all the history before that.
        # So all the information of what to search and what was critiqued is going to be here.
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
# partial method: to populate some already known placeholders, When we invoke this prompt template, then we want to plug in here the current date.
# And this will be computed only when we invoke this prompt template, and this will be when we invoke our agent
).partial(
    time=lambda: datetime.datetime.now().isoformat(),
)


# first instruction field, this will be plugged in into our prompt template,
# And this is what is going to make our LLM generate the first answer.
first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~100 word answer."
)

# first responder chain: is going to take our prompt template and it's going to pipe it into the LLM gpt
# But not before we bind the answer question object as a tool for the function calling and by providing  {tool_choice="AnswerQuestion"}
# This will force LLM to always use the answer question tool, thus grounding the response to the object that we want to receive
# And this is a cool technique where the grounding of the LLM also comes from the Pydantic object we created,
# So this way we are going to make LLM give us exactly the response that we want.
first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 100 words.
"""

# force the answer of LLM to be that kind of object (ReviseAnswer)
revisor = actor_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")


if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )
    chain = (
        first_responder_prompt_template
        | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
        | parser_pydantic
    )

    res = chain.invoke(input={"messages": [human_message]})
    print(res)


