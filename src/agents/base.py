"""Agent base class for the outreach tracker.

All agents inherit this. It handles the LLM tool-use loop so subclasses
just need to provide a system prompt and a list of tools.
"""

from dataclasses import dataclass
from typing import Any, Callable
from anthropic import Anthropic
import time


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable


@dataclass
class AgentResult:
    output: Any
    iterations: int
    tool_calls: list[dict]
    raw_messages: list[dict]
    elapsed_seconds: float
    success: bool
    error: str | None = None


class Agent:

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool] | None = None,
        model: str = "claude-sonnet-4-5",
        max_iterations: int = 10,
        max_tokens: int = 4096,
        client: Anthropic | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.model = model
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens
        self.client = client or Anthropic()

    def _tools_for_api(self) -> list[dict]:
        """Convert Tool objects to the format the Anthropic API expects."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools
        ]

    def _find_tool(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)

    def run(self, user_input: str) -> AgentResult:
        """Run the agent loop until it produces a final response or hits limits."""
        start = time.time()
        messages = [{"role": "user", "content": user_input}]
        tool_calls_log: list[dict] = []
        iterations = 0

        try:
            for _ in range(self.max_iterations):
                iterations += 1
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system_prompt,
                    tools=self._tools_for_api() if self.tools else [],
                    messages=messages,
                )

                # If the agent is done, return its text output
                if response.stop_reason == "end_turn":
                    output = self._extract_text(response)
                    return AgentResult(
                        output=output,
                        iterations=iterations,
                        tool_calls=tool_calls_log,
                        raw_messages=messages,
                        elapsed_seconds=time.time() - start,
                        success=True,
                    )

                # Handle tool use
                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})
                    tool_results = []

                    for block in response.content:
                        if block.type == "tool_use":
                            tool = self._find_tool(block.name)
                            if not tool:
                                result = f"Error: tool '{block.name}' not found"
                            else:
                                try:
                                    result = tool.handler(**block.input)
                                except Exception as e:
                                    result = f"Tool error: {e}"

                            tool_calls_log.append({
                                "tool": block.name,
                                "input": block.input,
                                "output": str(result)[:500],
                            })
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result),
                            })

                    messages.append({"role": "user", "content": tool_results})
                    continue

                # Unexpected stop reason — bail out
                break

            return AgentResult(
                output=None,
                iterations=iterations,
                tool_calls=tool_calls_log,
                raw_messages=messages,
                elapsed_seconds=time.time() - start,
                success=False,
                error="Max iterations reached",
            )

        except Exception as e:
            return AgentResult(
                output=None,
                iterations=iterations,
                tool_calls=tool_calls_log,
                raw_messages=messages,
                elapsed_seconds=time.time() - start,
                success=False,
                error=str(e),
            )

    @staticmethod
    def _extract_text(response) -> str:
        """Pull the text content out of an Anthropic response."""
        return "".join(b.text for b in response.content if hasattr(b, "text"))
