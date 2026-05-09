""" test: verify the Agent base class works end-to-end with a tool call."""

from dotenv import load_dotenv
from src.agents.base import Agent, Tool

load_dotenv()


def echo(message: str) -> str:
    """Test tool that echoes back its input."""
    return f"You said: {message}"


def test_agent_runs_with_tool():
    agent = Agent(
        name="test_agent",
        system_prompt=(
            "You are a test agent. When given a message, use the echo tool "
            "with that message, then respond with a brief summary of what "
            "the tool returned."
        ),
        tools=[
            Tool(
                name="echo",
                description="Echoes the given message back. Use this exactly once.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to echo",
                        },
                    },
                    "required": ["message"],
                },
                handler=echo,
            )
        ],
    )

    result = agent.run("Please echo the word 'hello'")

    assert result.success, f"Agent failed: {result.error}"

    assert any(tc["tool"] == "echo" for tc in result.tool_calls), \
        f"Expected echo tool call, got: {result.tool_calls}"

    assert isinstance(result.output, str) and len(result.output) > 0, \
        f"Expected text output, got: {result.output}"

    print(f"\n  Iterations: {result.iterations}")
    print(f"  Tool calls: {len(result.tool_calls)}")
    print(f"  Elapsed: {result.elapsed_seconds:.2f}s")
    print(f"  Output: {result.output[:200]}")


if __name__ == "__main__":
    test_agent_runs_with_tool()
    print("\n✓ Smoke test passed")
