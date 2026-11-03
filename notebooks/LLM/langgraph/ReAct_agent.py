from typing import Annotated,Sequence,TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END,START
from langgraph.prebuilt import ToolNode


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(num1:int,num2:int)->int:
    """This is an addition function that adds two numbers"""
    return num1+num2

tools  = [add]

llm = ChatOllama(
    model="llama3.2",
    temperature=0
).bind_tools(tools)

def model_call(state:AgentState)->AgentState:
    systemPrompt = SystemMessage(content="You are my AI assitant, please answer my query to the best of your ability")
    response = llm.invoke([systemPrompt]+state["messages"])
    return {"messages":[response]}

def should_continue(state:AgentState):
    messages = state['messages']
    last_message = messages[-1]
    if not last_message.tool_calls:
        return 'end'
    else:
        return 'continue'

graph = StateGraph(AgentState)
graph.add_node("our_agent",model_call)

tool_node = ToolNode(tools)
graph.add_node('tools',tool_node)
graph.add_edge(START,'our_agent')

graph.add_conditional_edges(
    'our_agent',should_continue,{
        'continue':'tools',
        'end': END
    }
)

graph.add_edge('tools','our_agent')
app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            
inputs = {"messages": [("user", "Add 40 + 12 Also tell me a joke please.")]}

print_stream(app.stream(inputs,stream_mode='values'))