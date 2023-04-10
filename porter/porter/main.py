from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.agents import (
    AgentExecutor,
    AgentType,
    Tool,
    ZeroShotAgent,
    initialize_agent,
)
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import BashProcess

prefix = """
Your task is to port js codebase to ts.
Codebase is in directory: /home/oskar/git/korczak-xyz/korczak-xyz/src.
You have to work in steps.
0. prefix all your bash commands with propper pwd
1. count all remaining js files. If count = 0 finish. If count is > 0 go to step 2
2. pick random js file from directory
3. change its extension to tsx
4. run npx tsc command to check if file is valid
5. if file is valid go to step 1
6. if file is invalid fix it and go to step 4
You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

llm = ChatOpenAI(temperature=0.0)
bash = BashProcess()
tools = [
    Tool(
        name="bash",
        func=bash.run,
        description="usefull for running bash commands",
    )
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


agent_chain.run(input="start")
