import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from archive.research import research_contact


def main():
    if len(sys.argv) < 3:
        print("Usage: uv run python run_research.py <name> <institution> [hint]")
        sys.exit(1)

    name = sys.argv[1]
    institution = sys.argv[2]
    hint = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"\n{'=' * 70}")
    print(f"Researching: {name} at {institution}")
    if hint:
        print(f"Hint: {hint}")
    print(f"{'=' * 70}\n")

    profile, result = research_contact(name, institution, hint=hint)

    # print agent metadata and tool calls for debugging and analysis
    print(f"\n--- Agent run ---")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls: {len(result.tool_calls)}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")
    if result.error:
        print(f"Error: {result.error}")

    print(f"\n--- Tool calls log ---")
    for i, tc in enumerate(result.tool_calls, 1):
        print(f"{i}. {tc['tool']}({tc['input']})")
        print(f"   → {tc['output'][:200]}{'...' if len(tc['output']) > 200 else ''}")

    print(f"\n--- Profile ---")
    if profile is None:
        print("[Parsing failed or agent failed — raw output:]")
        if result.output is None:
            print("(no output — agent failed before producing text)")
        else:
            print(result.output[:2000])

    # Save run to logs for later analysis
    log_dir = Path("logs/agent_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = name.replace(" ", "_").replace(".", "")
    log_path = log_dir / f"{timestamp}_research_{safe_name}.json"
    with log_path.open("w") as f:
        json.dump({
            "input": {"name": name, "institution": institution, "hint": hint},
            "profile": profile,
            "raw_output": result.output,
            "iterations": result.iterations,
            "tool_calls": result.tool_calls,
            "elapsed_seconds": result.elapsed_seconds,
            "success": result.success,
            "error": result.error,
        }, f, indent=2, default=str)
    print(f"\nLogged to: {log_path}")


if __name__ == "__main__":
    main()
