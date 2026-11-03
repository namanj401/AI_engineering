from langgraph.constants import END
from langgraph.constants import START
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from langchain_ollama.chat_models import ChatOllama
import traceback
from contextlib import redirect_stderr
from contextlib import redirect_stdout
import io
from langchain_core.tools import tool
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated
from typing import TypedDict,Sequence


class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage],add_messages]
    sender : str
    next: str
    
@tool
def python_repl(code:str):
    """Execute Python code in a local environment and return stdout or errors."""
    output_buffer = io.StringIO()
    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            exec(code)
        output = output_buffer.getvalue()
        return output if output else "Code executed successfully (no output to print)"
    except Exception:
        return f'Error executing code:\n{traceback.format_exc()}'
    
tools = [python_repl]

llm = ChatOllama(model='llama3.2:latest').bind_tools(tools)


def coder_node(state:AgentState)->AgentState:
    system_prompt = """You are an autonomous Python QA Engineer. Your objective is to fix buggy code and verify it.

CRITICAL DIRECTIVES:
1. NO CHIT-CHAT. Do not output any conversational text, step-by-step analysis, or explanations before writing code.
2. IMMEDIATE ACTION. You MUST immediately invoke the `python_repl` tool to execute and test your fixed code. 
3. VERIFICATION. Always include `print()` statements in the code you send to the tool to verify the output matches expected constraints.
4. ITERATION. If the tool returns an error, do not apologize. Immediately invoke the tool again with a corrected script.
5. COMPLETION. You are only allowed to output standard conversational text and the final Markdown code block AFTER the `python_repl` tool returns a successful execution verifying your fix.
"""
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message]+ list(state['messages'])
    response = llm.invoke(messages)
    
    return {
        'messages':[response],
        'sender': 'coder'
    }
    
def execution_node(state:AgentState)->AgentState:
    
    result = tool_node.invoke(state)
    
    result['sender'] = 'executioner'
    
    return result

tool_node = ToolNode(tools)


def supervisor_node(state:AgentState)->AgentState:
    last_message = state['messages'][-1]
    sender = state['sender']
    
    if sender is None or sender =='user' or isinstance(last_message,HumanMessage):
        return {'next':'coder'}
    
    if sender == 'executioner':
        return {'next':'coder'}
    
    if sender == 'coder':
        has_tool_calls = hasattr(last_message,'tool_calls') and len(last_message.tool_calls)>0
        if has_tool_calls:
            return {'next':'executioner'}
        else:
            return {'next':'finish'}
    
    return {'next':'coder'}

def route_next(state:AgentState)->str:
    return state['next']

graph = StateGraph(AgentState)

graph.add_node('supervisor',supervisor_node)
graph.add_node('coder_node',coder_node)
graph.add_node('executioner_node',execution_node)
graph.add_edge(START,'supervisor')

graph.add_conditional_edges('supervisor',route_next,{
    'coder':'coder_node',
    'executioner':'executioner_node',
    'finish' : END
})
graph.add_edge('coder_node','supervisor')
graph.add_edge('executioner_node','supervisor')
app = graph.compile()


# Define a buggy python script scenario
initial_buggy_prompt = (
    "Fix this Python function so it works correctly and runs without errors. "
    "It is supposed to return the first 'n' numbers of the Fibonacci sequence: "
    "\n\n"
    "def fibonacci(n):\n"
    "    if n <= 0:\n"
    "        return 'Error: Invalid input'\n"
    "    sequence = (0, 1) # This is a tuple, can we append to it? \n"
    "    while len(sequence) < n:\n"
    "        sequence.append(sequence[-1] + sequence[-2])\n"
    "    return sequence[:n]\n"
    "\n"
    "Make sure you run and verify your test code using python_repl first to "
    "verify that the output matches the expected result before finishing!"
)

# Initialize state
initial_state = {
    "messages": [HumanMessage(content=initial_buggy_prompt)],
    "sender": "user"
}

# Run it!
for event in app.stream(initial_state, {"recursion_limit": 50}):
    print("\n--- NEW EVENT ---")
    for node_name, output_state in event.items():
        print(f"[{node_name}] finished.")
        # print out messages generated in this node
        if "messages" in output_state and output_state["messages"]:
            last_msg = output_state["messages"][-1]
            print(f"Sender: {output_state.get('sender', 'unknown')}")
            print(f"Content:\n{last_msg.content[:500]}...") # truncate for visibility