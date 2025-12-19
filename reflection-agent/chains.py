# Import ChatPromptTemplate and MessagesPlaceholder from langchain_core.prompts
# - ChatPromptTemplate: A class used to create structured prompt templates for chat models
# - MessagesPlaceholder: A special placeholder that allows dynamic insertion of message history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv

# Import ChatOpenAI from langchain_openai
# - ChatOpenAI: A wrapper class for OpenAI's chat models (like GPT-3.5, GPT-4)
from langchain_openai import ChatOpenAI

load_dotenv()

# Create a prompt template for the reflection/critique agent
# This template defines how the AI should behave when evaluating tweets
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        # First message: System message that sets the AI's role and behavior
        # - "system": Indicates this is a system-level instruction (not from user or assistant)
        # - The text defines the AI as a viral Twitter influencer who grades tweets
        # - It instructs the AI to generate critiques and recommendations
        # - It emphasizes providing detailed feedback on length, virality, style, etc.
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),

        # Second element: MessagesPlaceholder for dynamic message history
        # - variable_name="messages": This is a placeholder that will be replaced with actual conversation messages
        # - Allows the prompt to include the full conversation context (previous tweets, critiques, etc.)
        # - The messages will be inserted here when the prompt is actually used
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Create a prompt template for the generation agent
# This template defines how the AI should behave when creating tweets
generation_prompt = ChatPromptTemplate.from_messages(
    [
        # System message defining the tweet generation role
        # - Sets the AI as a "twitter techie influencer assistant"
        # - Primary task: write excellent Twitter posts
        # - Instructs to generate the best possible tweet for user's request
        # - If critique is provided, revise the previous attempt accordingly
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),

        # MessagesPlaceholder for conversation history
        # - Same as above: allows dynamic insertion of the conversation context
        # - Will include user requests, previous tweet attempts, and any feedback
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize the language model (LLM)
# - ChatOpenAI(): Creates an instance of OpenAI's chat model
# - No parameters specified, so it uses defaults (typically GPT-3.5-turbo)
# - This is the actual AI model that will process the prompts
llm = ChatOpenAI(model='gpt-4o')

# Create the generation chain
# - Uses the pipe operator (|) to chain the generation_prompt with the llm
# - This creates a processing pipeline: prompt template -> LLM
# - When invoked, it will:
#   1. Format the generation_prompt with provided messages
#   2. Send the formatted prompt to the LLM
#   3. Return the LLM's response (a generated tweet)
generate_chain = generation_prompt | llm

# Create the reflection chain
# - Similar to generate_chain, but uses reflection_prompt
# - Creates pipeline: reflection prompt template -> LLM
# - When invoked, it will:
#   1. Format the reflection_prompt with provided messages
#   2. Send the formatted prompt to the LLM
#   3. Return the LLM's response (critique and recommendations)
reflect_chain = reflection_prompt | llm