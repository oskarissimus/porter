from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.utilities import BashProcess

prefix = """
Your task is to port js codebase to ts.
You have to work in steps.
1. count all remaining js files. If count = 0 finish. If count is > 0 go to step 2
2. pick random js file from directory
3. change its extension to tsx
4. validate file
5. if file is valid go to step 1
6. if file is invalid go to step 2
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
    ),
    Tool(
        name="count remaining js files",
        func=lambda x: bash.run("find . -name '*.js' | wc -l"),
        description="usefull for counting remaining js files",
    ),
    Tool(
        name="pick random js file",
        func=lambda x: bash.run("find . -name '*.js' | shuf -n 1"),
        description="usefull for picking random js file",
    ),
    Tool(
        name="validate file",
        func=lambda x: bash.run("npx tsc {x}"),
        description="usefull for validating file",
    ),
    Tool(
        name="show whats wrong with file",
        func=lambda x: bash.run("npx tsc {x} 2>&1 || exit 0"),
        description="usefull for showing whats wrong with file",
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


agent_chain.run(input="start")
