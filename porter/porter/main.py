from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import BashProcess

prefix = """
your task is to fix the typescript file.
you work in steps.
1. validate the file
2. fix the file
3. go to step 1
You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

llm = ChatOpenAI(temperature=0.0)
bash = BashProcess(return_err_output=True)
tools = [
    Tool(
        name="bash",
        func=bash.run,
        description="usefull for running bash commands",
    ),
    Tool(
        name="validate",
        func=lambda x: bash.run("npx tsc"),
        description="usefull for validating typescript files",
    ),
]

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)
memory = ConversationBufferMemory(memory_key="chat_history")


llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)
# commit on branch


agent_chain.run(input="your file to fix is ./pages/index.tsx")
