#!/usr/bin/env python3
# In[ ]:
# Example from: https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#setup
# modified to use ChatLlamaCpp


# In[ ]:
get_ipython().system('pip install langchain langchain-community langchain-core langgraph')


# In[ ]:
get_ipython().system('pip install huggingface-hub')


# In[ ]:
from huggingface_hub import hf_hub_download

#hf_gguf_model = ["bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf" ]
#hf_gguf_model = ["bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"]
hf_gguf_model = ["Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf"]
model_path = hf_hub_download(*hf_gguf_model)


# In[ ]:
get_ipython().system('pip install llama-cpp-python')


# In[ ]:
from langchain_community.chat_models import ChatLlamaCpp

model = ChatLlamaCpp(
    model_path=model_path,
    verbose=False,
    stop=["<|end_of_text|>", "<|eot_id|>"],
)


# In[ ]:
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from datetime import datetime

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    if location.lower() in ["sf", "san francisco"]:
        return "60 degrees and foggy."
    else:
        return "80 degrees and sunny."

@tool
def get_current_time() -> str:
    """This tool returns the current time."""
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return current_time

tools = [get_weather, get_current_time]

model_with_tools = model.bind_tools(tools)
tool_node = ToolNode(tools)


# In[ ]:
import re
import uuid
import json
from langgraph.graph import MessagesState

def json_strip(s):
    count = 0
    start = None
    end = None

    for i, c in enumerate(s):
        if c == '{': count += 1
        else:
            if start is None and count:
                count = 1
                start = i - 1
            if c == '}': count -= 1
        if count == 0:
            end = i + 1

    if end and start is not None: s = s[start:end]
    return s

def json_tool_call(s):
    s = json_strip(s.replace('\\"', '"').replace("'", '"').replace('}"', '}').replace('"{', '{'))
    tool_call_data = json.loads(s)
    tool_params = tool_call_data.get("arguments", tool_call_data.get("parameters", {}).get("properties", tool_call_data.get("parameters", {})))
    return {
        "name": tool_call_data["name"],
        "args": tool_params,
        "id": str(uuid.uuid4()),
        "type": "tool_call"
    }

def fix_tool_calls(state: MessagesState):
    if ( state.tool_calls is not None and 0 != len(state.tool_calls)) or state.content is None: return state

    content = state.content.strip()
    m = re.findall(r'<tool_call>\s*(\{.*\})\s*</tool_call>', content, re.DOTALL)
    if m is None or 0 == len(m):
        m = re.findall(r'<tool_call>\s*(\{.*\})', content, re.DOTALL)
    try:
        if m is not None and len(m) > 0:
            state.tool_calls = [json_tool_call(s) for s in m]
            state.content = ""
        elif content[0] == '{':
            state.tool_calls = [ json_tool_call(content) ]
            state.content = ""
    except json.JSONDecodeError as e:
        print(str(e) + content)
    return state


# In[ ]:
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import tools_condition

def call_model(state: MessagesState):
    messages = state["messages"]
    response = fix_tool_calls(model_with_tools.invoke(messages))
    return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", tools_condition, ["tools", END])
workflow.add_edge("tools", "agent")

app = workflow.compile()


# In[ ]:
for chunk in app.stream(
    {"messages": [("human", "what is the current time?")]}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()

for chunk in app.stream(
    {"messages": [("human", "what is the weather in SF?")]}, stream_mode="values"
):
    chunk["messages"][-1].pretty_print()
