# example from: https://api.python.langchain.com/en/latest/agents/langchain.agents.react.agent.create_react_agent.html
# with tools from https://python.langchain.com/docs/how_to/output_parser_string/
# modified to use ChatLlamaCpp
# https://python.langchain.com/v0.2/api_reference/community/chat_models/langchain_community.chat_models.llamacpp.ChatLlamaCpp.html
# example updated with dummy image tools

from langchain_community.chat_models import ChatLlamaCpp
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain.callbacks.manager import CallbackManager

from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.callbacks.base import BaseCallbackHandler

from huggingface_hub import hf_hub_download

#hf_gguf_model = ["bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf" ]
#hf_gguf_model = ["bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"]
hf_gguf_model = ["bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"]
model_path = hf_hub_download(*hf_gguf_model)

model = ChatLlamaCpp(
    model_path=model_path,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
    seed=1,
)

@tool(parse_docstring=True)
def denoise_image(image_name: str) -> str:
    """Denoise an image

    Args:
        image_name: image file name

    Returns:
        output image file name
    """
    return 'dn-' + image_name

@tool(parse_docstring=True)
def unblur_image(image_name: str) -> str:
    """Unblur an image

    Args:
        image_name: image file name

    Returns:
        output image file name
    """
    return 'ub-' + image_name

@tool(parse_docstring=True)
def check_image_blur(image_name: str) -> dict:
    """Find the blur the an image

    Args:
        image_name: image file name

    Returns:
        image blur
    """
    blur = (48 - (len(image_name) % 48))/48
    return { 'blur': blur, 'blur_acceptable': blur < 0.7 }

tools = [ denoise_image, unblur_image, check_image_blur ]

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: give only the name of tool to use, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the brief final output satisfying the original input question

Begin!

Question: {input}
{agent_scratchpad}'''

class CustomHandler(BaseCallbackHandler):
    def on_tool_end(self, output, **kwargs):
        print(f"\nObservation: {output}")

prompt = PromptTemplate.from_template(template)

agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, callbacks=[CustomHandler()]) #, return_intermediate_steps=True)

question = "First, denoise the image named test33.jpg. Next, check the blur level of the image. If the blur value is not acceptable, unblur the image. Repeat the unblur image process until the blur checks to be acceptable. Once the image blur is acceptable, as a final answer, give just the final image file name along with the blur value of the image."

print(f"Question: {question}")

agent_executor.invoke({ "input": question }, { 'callbacks':[CustomHandler()] })
