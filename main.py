from __future__ import annotations

import warnings
import requests

from langgraph.prebuilt import create_react_agent
from langGraph_foundry import VantageAPIConnector, PalantirFoundryChatModel
from requests.packages.urllib3.exceptions import InsecureRequestWarning


connector = VantageAPIConnector(
    bearer_token="vantage token"
)

agent1 = connector.create_agent(
    agent_name="agent_1",
    agent_id="agentRID",
    version="8.0",
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

def get_forecast(city: str) -> str:
    """Get the weather forecast for the next week for a given city."""
    return f"There is a hurricane all of next week in {city}!"

def get_advice() -> str:
    """This function gets advice from an API for the user."""
    try:
        res = requests.get("https://api.adviceslip.com/advice", verify=False)
        if res.status_code == 200:
            advice = res.json().get('slip', {}).get('advice', "No advice available")
            print(f"DEBUG - API returned advice: {advice}")
            return advice
        else:
            print(f"DEBUG - API returned status code: {res.status_code}")
            return f"Sorry, couldn't get advice. Status code: {res.status_code}"
    except Exception as e:
        print(f"DEBUG - Error in get_advice: {str(e)}")
        return f"Error getting advice: {str(e)}"



palantir_llm = PalantirFoundryChatModel(agent=agent1, max_iterations=2)
palantir_llm_with_tools = palantir_llm.bind_tools([get_weather, get_forecast, get_advice])

# Create a consistent prompt
tool_prompt = """
You are a helpful assistant and fruit fanatic. All of your responses should mention fruit in them. You have access to the following tools:
get_weather(city: str) - Get weather for a given city
get_forecast(city: str) - Get weather forecast for the next week for a given city
get_advice() - Get the user some friendly advice from an internet API

IMPORTANT: After receiving the result of a tool, you MUST use EXACTLY that result to answer the user's question. 
Do NOT make up additional information or ignore the tool's output.

When you need to use a tool, use the following format:
Thought: think about what to do
Action: get_weather
Action Input: {"city": "city name"}
Observation: the result of the action
Final Answer: the final answer based EXACTLY on the observation
"""



# Create the agent
palantir_agent_node = create_react_agent(
    model=palantir_llm_with_tools,  # Use the model with tools already bound
    tools=[get_weather, get_forecast, get_advice],
    prompt=tool_prompt,
    name="palantir_agent",
)

# Now try the regular agent invocation
print("\n--- Testing agent invocation ---")
resp_state = palantir_agent_node.invoke(
    {"messages": [{"role": "user", "content": "can you give me some advice?"}]} #what is the weather like in austin?
    #{"messages": [{"role": "user", "content": "what is the weather like in austin?"}]}
    #{"messages": [{"role": "user", "content": "what is the weather going to be like in austin?"}]}
)
print(resp_state)