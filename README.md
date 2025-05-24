# langgraph_foundry - Palantir Foundry AIP agent wrapper to langgraph 

A lightweight utility that lets you **plug Palantir Foundry AIP agents straight into LangChain graphs or nodes**.  
It wraps the Palantir “blockingContinue” session API in a LangChain‑compatible `BaseChatModel`, adds a mini ReAct loop, and lets you bind ordinary Python functions as tools.

---

## ✨ Features

| Capability | Details |
|------------|---------|
| **Session Handling** | Automatically starts and caches a Foundry AIP session (RID) for each agent. |
| **ReAct‑style Tool Use** | Parses `Action:` / `Action Input:` blocks and executes bound tools locally, then feeds observations back to the agent. |
| **LangChain Compatible** | Exposes `_generate` so you can drop it into LangChain graphs (e.g. `create_react_agent`). |
| **Flexible Tool Binding** | Bind plain callables or LangChain `BaseTool` instances via `bind_tools([...])`. |
| **Debug Friendly** | Emits debug prints (optional removal) and suppresses `urllib3` insecure‑request warnings for dev convenience. |

---

## 📦 Installation

```bash
git clone https://github.com/jameswm2/langgraph_foundry.git
cd langgraph_foundry
conda create -n langgraph_foundry
conda activate langgraph_foundry
pip install -r requirements.txt   # langchain, langgraph, pydantic, requests …
```

```python
from langGraph_foundry import VantageAPIConnector, PalantirFoundryChatModel
from langgraph.prebuilt import create_react_agent

# 1  Connect to Foundry
connector = VantageAPIConnector(bearer_token="YOUR_BEARER_TOKEN")
agent = connector.create_agent(
    agent_name="support‑bot",
    agent_id="ri.aip-agents..agent.<uuid>",
    version="8.0",
)

# 2  Wrap in LangChain model
llm = PalantirFoundryChatModel(agent=agent, max_iterations=2)

# 3  Define a simple tool
def get_weather(city: str) -> str:
    return f"It’s always sunny in {city}!"

# 4  Bind the tool
llm_with_tools = llm.bind_tools([get_weather])

# 5  Create a ReAct agent
prompt = """You are a helpful assistant and fruit fanatic …"""
react_agent = create_react_agent(
    model=llm_with_tools,
    tools=[get_weather],
    prompt=prompt,
    name="palantir_agent",
)

# 6  Invoke
resp = react_agent.invoke(
    {"messages": [{"role": "user", "content": "What’s the weather in Austin?"}]}
)
print(resp)
```
