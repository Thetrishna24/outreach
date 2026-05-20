import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.agents.faculty_finder import find_faculty


def main():
    if len(sys.argv) < 3:
        print("Usage: uv run python scout.py <university> <field1> [field2] ...")
        print("Example: uv run python scout.py 'Harvard University' 'bioinformatics' 'computational biology'")
        sys.exit(1)

    university = sys.argv[1]
    fields = sys.argv[2:]

    print(f"\n{'=' * 70}")
    print(f"University: {university}")
    print(f"Fields: {', '.join(fields)}")
    print(f"{'=' * 70}\n")

    result_dict, result = find_faculty(university, fields)

    print(f"\n--- Agent run ---")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.iterations}")
    print(f"Tool calls: {len(result.tool_calls)}")
    print(f"Elapsed: {result.elapsed_seconds:.2f}s")
    if result.error:
        print(f"Error: {result.error}")

    print(f"\n--- Tool calls ---")
    for i, tc in enumerate(result.tool_calls, 1):
        print(f"{i}. {tc['tool']}({tc['input']})")
        out = tc['output']
        print(f"   → {out[:150]}{'...' if len(out) > 150 else ''}")

    print(f"\n--- Result ---")
    if result_dict is None:
        print("[Parsing failed]")
        if result.output:
            print(result.output[:2000])
    else:
        print(json.dumps(result_dict, indent=2))

    # Save run log
    log_dir = Path("logs/agent_runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_uni = university.replace(" ", "_").replace("/", "_")
    log_path = log_dir / f"{timestamp}_faculty_{safe_uni}.json"
    with log_path.open("w") as f:
        json.dump({
            "input": {"university": university, "fields": fields},
            "result": result_dict,
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
