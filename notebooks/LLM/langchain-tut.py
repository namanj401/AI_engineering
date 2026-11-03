from langchain_core.tools import tool
from langchain.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import wikipedia
from dotenv import load_dotenv

@tool
def add_numbers(x: int, y: int) -> int:
    """Adds two numbers and returns the sum."""
    return x + y

wikipedia.set_user_agent("LangChainTut/1.0 (test@example.com)")
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
messages = [HumanMessage(content="What was the most impressive thing about Buzz Aldrin?")]
llm_with_tools = llm.bind_tools([tool])
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
   tool_msg = tool.invoke(tool_call)
   print(tool_msg.name)
   print(tool_call['args'])
   print(tool_msg.content)
   messages.append(tool_msg)
   print()

final_response = llm_with_tools.invoke(messages)
print(final_response.content)





