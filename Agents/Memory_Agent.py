# Memory Agent using LangGraph and LangChain
import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# load_dotenv()
# Load the environment variables from the .env file
os.environ["OPENAI_API_KEY"] = "api_key"
load_dotenv(dotenv_path=".env") # Load environment variables from .env file
print("API KEY:", os.getenv("OPENAI_API_KEY"))

# This is a simple agent that uses the OpenAI API to respond to user input.
# It uses a state graph to manage the flow of conversation.
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# The agent uses the ChatOpenAI class from the langchain_openai module to interact with the OpenAI API.
# The ChatOpenAI class is initialized with the model name "gpt-4o".
llm = ChatOpenAI(model="gpt-4o")

# The process function is called when the agent is invoked.
# It takes the current state of the agent as input and returns the updated state.
def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) 
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])

    return state

# The agent is created using the StateGraph class from the langgraph.graph module.
# The StateGraph class is initialized with the AgentState type.
# The agent starts with a message from the user, and then it responds with a message from the AI.
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()


# The agent is invoked with an initial message from the user.
print("Welcome to the Memory Agent! Type 'exit' to quit.")
conversation_history = []

# Initialize the conversation history with a system message
user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

# Save the conversation history to a text file
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")