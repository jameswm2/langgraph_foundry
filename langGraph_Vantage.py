import json
import re
from typing import List, Optional

import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, ChatGeneration, ChatResult
from pydantic import Field
from langchain_core.tools import BaseTool
from langchain.tools import tool

class VantageAPIConnector:
    """Connects to the Vantage API."""

    def __init__(
        self,
        bearer_token: str,
        base_url: str = "https://vantage.army.mil/api/v2",
    ):
        """
        Initializes the Vantage API Connector.

        Args:
            bearer_token: The bearer token for authentication.
            base_url: The base URL of the Vantage API.  Defaults to the production URL.
        """
        self.bearer_token = bearer_token
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/58.0.3029.110 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
        }

    def create_agent(
        self,
        agent_name: str,
        agent_id: str,
        version: str,
    ) -> "VantageAPIConnector.Agent":
        """
        Creates an Agent instance.

        Args:
            agent_name: The name of the agent.
            agent_id: The ID of the agent.
            version: The version of the agent.

        Returns:
            An Agent instance.
        """
        return VantageAPIConnector.Agent(self, agent_name, agent_id, version)

    class Agent:
        """Represents a Vantage AIP Agent."""

        def __init__(
            self,
            connector: "VantageAPIConnector",
            agent_name: str,
            agent_id: str,
            version: str,
        ):
            """
            Initializes an Agent instance.

            Args:
                connector: The VantageAPIConnector instance.
                agent_name: The name of the agent.
                agent_id: The ID of the agent.
                version: The version of the agent.
            """
            self.connector = connector
            self.agent_name = agent_name
            self.agent_id = agent_id
            self.version = version
            self.session: Optional[str] = None

        def start_session(self) -> str:
            """
            Starts a new Palantir AIP session and caches the session ID.

            Returns:
                The session ID (rid).

            Raises:
                RuntimeError: If the session cannot be started or the response
                    doesn't contain a session ID.
            """
            url = (
                f"{self.connector.base_url}"
                f"/aipAgents/agents/{self.agent_id}/sessions"
            )
            params = {"preview": "true"}
            payload = {"agentVersion": self.version}

            try:
                res = requests.post(
                    url,
                    headers=self.connector.headers,
                    params=params,
                    json=payload,
                    verify=False,  # ‼︎ remove in production
                    timeout=30,
                )
                res.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Unable to start session for agent {self.agent_id}: {exc}"
                ) from exc

            response_json = res.json()
            self.session = response_json.get("rid")

            if not self.session:
                raise RuntimeError(
                    f"Vantage response did not contain a session rid – body was: {res.text}"
                )
            return self.session

        def send_query(self, text: str) -> dict:
            """
            Sends a user query to the Vantage AIP agent.

            Args:
                text: The user's query.

            Returns:
                The JSON response from the Vantage API.

            Raises:
                RuntimeError: If the query fails.
            """
            if self.session is None:
                self.start_session()

            url = (
                f"{self.connector.base_url}"
                f"/aipAgents/agents/{self.agent_id}/sessions/{self.session}"
                "/blockingContinue"
            )
            params = {"preview": "true"}
            payload = {"userInput": {"text": text}}

            try:
                res = requests.post(
                    url,
                    headers=self.connector.headers,
                    params=params,
                    json=payload,
                    verify=False,  # ‼︎ remove in production
                    timeout=120,
                )
                res.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(
                    f"Query failed (status {res.status_code}): {res.text}"
                ) from exc

            return res.json()

ACTION_RE = re.compile(r"Action:\s*([\w_]+)", re.I | re.M)  # More flexible for tool names
AINPUT_RE = re.compile(r"Action\s+Input:\s*(\{.+?\})", re.I | re.S | re.DOTALL)  # Better JSON capture
FINAL_RE = re.compile(r"Final\s+Answer:\s*(.*?)(?:\n\n|$)", re.I | re.S | re.DOTALL)


class PalantirFoundryChatModel(BaseChatModel):
    """Wrapper around a Palantir AIP agent with an *internal* ReAct loop,
    modeled after OpenAI's tool-using behavior."""

    agent: "VantageAPIConnector.Agent" = Field(...)  # Type hint as a string to avoid circular dependency
    max_iterations: int = Field(default=3)

    # ───────────────────────────────────────────────────────────── LLM type
    @property
    def _llm_type(self) -> str:
        return "palantir_foundry"

    # ───────────────────────────────────────────────────────────── Tools
    def bind_tools(self, tools: List) -> "PalantirFoundryChatModel":
        """Attach tools so the wrapper can execute them locally."""
        normalised: List[BaseTool] = []
        for t in tools:
            if isinstance(t, BaseTool):
                normalised.append(t)
            elif callable(t):
                normalised.append(tool()(t))
            else:
                raise TypeError(f"Tool {t!r} is neither BaseTool nor callable")
        clone = self.model_copy()
        object.__setattr__(clone, "_bound_tools", {tl.name: tl for tl in normalised})
        return clone

    # Helper function to extract tool information from text
    def _extract_tool_info(self, text):
        """Extract tool name, input, and final answer from text using a line-by-line approach."""
        lines = text.split('\n')
        tool_name = None
        tool_input_str = None
        final_answer = None
        
        # First pass: check for tool call
        for line in lines:
            if line.strip().startswith('Action:'):
                tool_name = line.replace('Action:', '').strip()
                print(f"DEBUG: Detected tool call: {tool_name}")
                break  # Found a tool call, prioritize this over final answer
                
        # If we found a tool call, look for its input
        if tool_name:
            for line in lines:
                if line.strip().startswith('Action Input:'):
                    tool_input_str = line.replace('Action Input:', '').strip()
                    print(f"DEBUG: Raw tool input: {tool_input_str}")
                    break
                    
            # Don't look for final answer if we found a tool call
            final_answer = None
        else:
            # Only look for final answer if no tool call was found
            for i, line in enumerate(lines):
                if line.strip().startswith('Final Answer:'):
                    final_answer = ' '.join([l.strip() for l in lines[i:]])
                    final_answer = final_answer.replace('Final Answer:', '').strip()
                    
                    # Check if it's a placeholder or empty answer
                    if '[' in final_answer and ']' in final_answer and 'will be provided' in final_answer:
                        print(f"DEBUG: Found placeholder final answer, ignoring: {final_answer}")
                        final_answer = None
                    else:
                        print(f"DEBUG: Found final answer: {final_answer[:100]}...")
                    break
        
        # Parse tool input
        tool_input = {}
        if tool_input_str:
            try:
                # Handle both JSON format and key-value format
                if tool_input_str.startswith('{') and tool_input_str.endswith('}'):
                    tool_input = json.loads(tool_input_str)
                else:
                    # Try to parse "city: SF" format
                    parts = tool_input_str.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        tool_input = {key: value}
                print(f"DEBUG: Parsed tool input: {tool_input}")
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON decode error: {e}")
                # Try a more lenient approach
                if "city" in tool_input_str.lower():
                    city_match = re.search(r'city[\":\s]+([^\"}\s]+)', tool_input_str, re.I)
                    if city_match:
                        tool_input = {"city": city_match.group(1)}
                        print(f"DEBUG: Extracted city using regex: {tool_input}")
        
        return tool_name, tool_input, final_answer

    # ───────────────────────────────────────────────────────────── Generate
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        # 0) ensure AIP session
        if self.agent.session is None:
            self.agent.start_session()

        # 1) prepare running transcript (list of strings)
        def _role(m: BaseMessage) -> str:
            return {"system": "System",
                    "assistant": "Assistant",
                    "ai": "Assistant",
                    "human": "User",
                    "user": "User"}.get(getattr(m, "role", "user"), "User")

        transcript: List[str] = [
            f"{_role(m)}: {m.content}" for m in messages
        ]

        # 2) iterate up to N reasoning/tool steps
        for step in range(self.max_iterations):
            print(f"\nDEBUG: Starting step {step+1}")
            prompt_text = "\n\n".join(transcript)
            raw = self.agent.send_query(prompt_text)

            reply = (
                raw.get("agentMarkdownResponse")
                or raw.get("text")
                or str(raw)
            )
            print(f"DEBUG: Raw reply: {reply[:100]}...")

            # Extract tool information using the helper function
            tool_name, tool_input, final_answer = self._extract_tool_info(reply)

            # If we find a valid final answer, return it
            if final_answer:
                ai_msg = AIMessage(content=final_answer)
                return ChatResult(generations=[ChatGeneration(message=ai_msg)])

            # Check if this is a tool call
            if tool_name and hasattr(self, "_bound_tools"):
                # Check if tool exists
                normalized_tool_name = tool_name.lower()
                available_tools = list(self._bound_tools.keys())
                print(f"DEBUG: Available tools: {available_tools}")

                # Try to match the tool name with available tools
                matching_tool = None
                for available_tool in available_tools:
                    if normalized_tool_name == available_tool.lower() or available_tool.lower() in normalized_tool_name:
                        matching_tool = available_tool
                        break

                if matching_tool:
                    print(f"DEBUG: Matched tool name '{tool_name}' to '{matching_tool}'")
                    # Execute the tool
                    try:
                        tool_instance = self._bound_tools[matching_tool]
                        print(f"DEBUG: About to execute tool {matching_tool}")
                        
                        if hasattr(tool_instance, 'invoke'):
                            if tool_input:
                                observation = tool_instance.invoke(tool_input)
                            else:
                                # Special case for zero-parameter functions
                                observation = tool_instance.invoke()
                        else:
                            # Fallback to the deprecated __call__ method
                            if tool_input:
                                observation = tool_instance(**tool_input)
                            else:
                                # Special case for zero-parameter functions
                                observation = tool_instance()
                                
                        print(f"DEBUG: Tool execution result: {observation}")

                    except Exception as e:
                        observation = f"Tool execution error: {e}"
                        print(f"DEBUG: Tool execution error: {e}")
                        last_observation = observation

                    # Append the assistant's reply and the observation to the transcript
                    transcript.append(f"Assistant: {reply}")
                    transcript.append(f"Observation: {observation}")

                    # **Crucial Change:** Instead of a separate prompt, force the final answer in the SAME turn
                    # This is how OpenAI's tool-using agents typically work.  We *expect* the agent to
                    # immediately incorporate the observation into a final answer.

                    # Modify the prompt to instruct the agent to use the tool result
                    final_answer_prompt = (
                        f"Based on the observation that {observation}, what is the final answer? "
                        f"Remember to include a fruit reference."
                        f"Do not make up additional details."
                    )
                    transcript.append(f"System: {final_answer_prompt}")

                    # Get one more response to generate the final answer
                    prompt_text = "\n\n".join(transcript)
                    raw = self.agent.send_query(prompt_text)
                    final_reply = (
                        raw.get("agentMarkdownResponse")
                        or raw.get("text")
                        or str(raw)
                    )

                    # Return this as the final answer
                    ai_msg = AIMessage(content=final_reply)
                    return ChatResult(generations=[ChatGeneration(message=ai_msg)])

                    # **End Crucial Change**

                    # Continue with the next iteration (this is likely not needed anymore)
                    #continue

            # Add the reply to the transcript for the next iteration
            transcript.append(f"Assistant: {reply}")

        # 4) ultimate fall‑back: just emit the most recent assistant text
        ai_msg = AIMessage(content=reply)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    # ───────────────────────────────────────────────────────────── Generate
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> ChatResult:
        # 0) ensure AIP session
        if self.agent.session is None:
            self.agent.start_session()

        # 1) prepare running transcript (list of strings)
        def _role(m: BaseMessage) -> str:
            return {"system": "System",
                    "assistant": "Assistant",
                    "ai": "Assistant",
                    "human": "User",
                    "user": "User"}.get(getattr(m, "role", "user"), "User")

        transcript: List[str] = [
            f"{_role(m)}: {m.content}" for m in messages
        ]

        # 2) iterate up to N reasoning/tool steps
        for step in range(self.max_iterations):
            print(f"\nDEBUG: Starting step {step+1}")
            prompt_text = "\n\n".join(transcript)
            raw = self.agent.send_query(prompt_text)

            reply = (
                raw.get("agentMarkdownResponse")
                or raw.get("text")
                or str(raw)
            )
            print(f"DEBUG: Raw reply: {reply[:100]}...")

            # Extract tool information using the helper function
            tool_name, tool_input, final_answer = self._extract_tool_info(reply)

            # If we find a valid final answer, return it
            if final_answer:
                ai_msg = AIMessage(content=final_answer)
                return ChatResult(generations=[ChatGeneration(message=ai_msg)])

            # Check if this is a tool call
            if tool_name and hasattr(self, "_bound_tools"):
                # Check if tool exists
                normalized_tool_name = tool_name.lower()
                available_tools = list(self._bound_tools.keys())
                print(f"DEBUG: Available tools: {available_tools}")

                # Try to match the tool name with available tools
                matching_tool = None
                for available_tool in available_tools:
                    if normalized_tool_name == available_tool.lower() or available_tool.lower() in normalized_tool_name:
                        matching_tool = available_tool
                        break

                if matching_tool:
                    print(f"DEBUG: Matched tool name '{tool_name}' to '{matching_tool}'")
                    # Execute the tool
                    try:
                        tool_instance = self._bound_tools[matching_tool]
                        if hasattr(tool_instance, 'invoke'):
                            observation = tool_instance.invoke(tool_input)
                        else:
                            # Fallback to the deprecated __call__ method
                            observation = tool_instance(**tool_input)
                        print(f"DEBUG: Tool execution result: {observation}")

                    except Exception as e:
                        observation = f"Tool execution error: {e}"
                        print(f"DEBUG: Tool execution error: {e}")
                        last_observation = observation

                    # Append the assistant's reply and the observation to the transcript
                    transcript.append(f"Assistant: {reply}")
                    transcript.append(f"Observation: {observation}")

                    # **Crucial Change:** Instead of a separate prompt, force the final answer in the SAME turn
                    # This is how OpenAI's tool-using agents typically work.  We *expect* the agent to
                    # immediately incorporate the observation into a final answer.

                    # Modify the prompt to instruct the agent to use the tool result
                    final_answer_prompt = (
                        f"Based on the observation that {observation}, what is the final answer? "
                        f"Remember to include a fruit reference."
                        f"Do not make up additional details."
                    )
                    transcript.append(f"System: {final_answer_prompt}")

                    # Get one more response to generate the final answer
                    prompt_text = "\n\n".join(transcript)
                    raw = self.agent.send_query(prompt_text)
                    final_reply = (
                        raw.get("agentMarkdownResponse")
                        or raw.get("text")
                        or str(raw)
                    )

                    # Return this as the final answer
                    ai_msg = AIMessage(content=final_reply)
                    return ChatResult(generations=[ChatGeneration(message=ai_msg)])

                    # **End Crucial Change**

                    # Continue with the next iteration (this is likely not needed anymore)
                    #continue

            # Add the reply to the transcript for the next iteration
            transcript.append(f"Assistant: {reply}")

        # 4) ultimate fall‑back: just emit the most recent assistant text
        ai_msg = AIMessage(content=reply)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])
