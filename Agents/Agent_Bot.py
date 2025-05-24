from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values
import os
os.environ["OPENAI_API_KEY"] = "api_key" # Set your OpenAI API key here
load_dotenv(dotenv_path=".env") # Load environment variables from .env file
print("API KEY:", os.getenv("OPENAI_API_KEY"))  # Check if key is loading

# This is a simple agent that uses the OpenAI API to respond to user input.
# It uses a state graph to manage the flow of conversation.
# The agent starts with a message from the user, and then it responds with a message from the AI.
# The agent continues to respond to user input until the user types "exit".
# The agent uses the ChatOpenAI class from the langchain_openai module to interact with the OpenAI API.
class AgentState(TypedDict):
    messages: List[HumanMessage]

# The agent uses the ChatOpenAI class from the langchain_openai module to interact with the OpenAI API.
# The ChatOpenAI class is initialized with the model name "gpt-4o".
# The model name can be changed to use a different model.
llm = ChatOpenAI(model="gpt-4o")

# The process function is called when the agent is invoked.
# It takes the current state of the agent as input and returns the updated state.
# The process function uses the ChatOpenAI class to generate a response to the user's message.
# The response is printed to the console, and the state is returned.

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

# The agent is created using the StateGraph class from the langgraph.graph module.
# The StateGraph class is initialized with the AgentState type.
# The agent starts with a message from the user, and then it responds with a message from the AI.
# The agent continues to respond to user input until the user types "exit".
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

# The agent is invoked with an initial message from the user.
# The user is prompted to enter a message, and the agent responds with a message from the AI.
print("Welcome to the Agent Bot! Type 'exit' to quit.")
messages = []
user_input = input("Enter: ")
while user_input != "exit":
    messages.append(HumanMessage(content=user_input))
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")