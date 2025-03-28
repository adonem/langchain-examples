# example from: https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
# with tools from https://python.langchain.com/docs/how_to/output_parser_string/
# modified to use ChatLlamaCpp
# https://python.langchain.com/v0.2/api_reference/community/chat_models/langchain_community.chat_models.llamacpp.ChatLlamaCpp.html

from langchain_community.chat_models import ChatLlamaCpp
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain.callbacks.manager import CallbackManager

from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from huggingface_hub import hf_hub_download

#hf_gguf_model = ["bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf" ]
#hf_gguf_model = ["bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"]
hf_gguf_model = ["bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"]
model_path = hf_hub_download(*hf_gguf_model)

model = ChatLlamaCpp(
    model_path=model_path,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
) 

@tool
def get_weather(location: str) -> str:
    """Get the weather from a location."""

    return "Sunny."

tools = [get_weather]

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke(
    {
        "input": "What's the weather in San Francisco, CA?"
    }
)
