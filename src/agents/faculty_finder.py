import json
from anthropic import Anthropic

from src.agents.base import Agent, Tool, AgentResult
from src.tools.web import web_fetch


SYSTEM_PROMPT = """You are a faculty research agent. Your job: given a US university and one or more research fields, find all professors at that university whose primary work falls in those fields.

Your process:
1. Search the web to find the relevant department(s) at the university for the given fields.
   - "Bioinformatics" → biology, computer science, biomedical engineering departments
   - "Computational neuroscience" → neuroscience, biology, psychology, brain & cognitive sciences
   - "Artificial Intelligence/Machine learning" → computer science, electrical engineering, statistics
2. Find the faculty list page(s) for those departments.
3. Fetch the faculty list page and extract every professor whose stated research area matches at least one of the requested fields.
4. For each matching professor, find:
   - Full name
   - Title (Professor, Associate Professor, Assistant Professor, etc.)
   - Department
   - Research focus (1-2 sentences in their own words if possible)
   - Profile URL (their faculty page)
   - Lab URL (their lab website if they have one, otherwise null)

Critical rules:
- Only include professors whose work CLEARLY matches the requested fields. When unsure, exclude rather than include.
- NEVER fabricate names, research focus, or URLs. If you don't have a source for it, leave it out or set to null.
- Every URL must be one you actually fetched or saw in search results.
- If you can't find a lab URL, set lab_url to null. Don't guess.
- If a professor's profile page is on the department site (e.g., utaustin.edu/...), that's the profile_url. The lab_url is a separate site (often tianhonglab.org-style).

Output format: When you have a complete list, respond with ONLY a JSON object matching this schema. No preamble, no markdown fences, no text after the JSON.

{
  "university": "string",
  "fields_searched": ["string", ...],
  "departments_explored": ["string", ...],
  "professors": [
    {
      "name": "string",
      "title": "string",
      "department": "string",
      "research_focus": "string",
      "profile_url": "string",
      "lab_url": "string or null",
      "matched_fields": ["string", ...]
    }
  ],
  "total_found": integer,
  "notes": "string (1-2 sentences about coverage — which departments you searched, any limitations)"
}

The professors array can be empty if no matches were found. Always include all required fields. Use null for missing optional values.
"""


def build_faculty_finder_agent(client: Anthropic | None = None) -> Agent:
    tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 10,
        },
        Tool(
            name="web_fetch",
            description=(
                "Fetch the full text content of a specific URL. "
                "Use after web_search to read faculty list pages and profile pages. "
                "Returns cleaned text truncated to 3000 characters."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL to fetch, including https://",
                    },
                },
                "required": ["url"],
            },
            handler=web_fetch,
        ),
    ]

    return Agent(
        name="faculty_finder",
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        model="claude-haiku-4-5",
        max_iterations=15,
        max_tokens=8192,
        client=client,
    )


def find_faculty(
    university: str,
    fields: list[str],
    client: Anthropic | None = None,
) -> tuple[dict | None, AgentResult]:
    """Find faculty at a university matching given research fields.

    Returns (parsed_result_dict, agent_result).
    parsed_result_dict is None if JSON parsing failed.
    """
    agent = build_faculty_finder_agent(client=client)

    fields_str = ", ".join(fields)
    user_input = (
        f"Find all professors at {university} whose primary research "
        f"falls in any of these fields: {fields_str}.\n\n"
        f"Return the structured JSON as specified."
    )

    result = agent.run(user_input)

    if not result.success or result.output is None:
        return None, result

    raw = result.output.strip()

    # Try parsing as-is
    try:
        return json.loads(raw), result
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            cleaned = part.strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
            if cleaned.startswith("{"):
                try:
                    return json.loads(cleaned), result
                except json.JSONDecodeError:
                    continue

    # Find first { and last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1]), result
        except json.JSONDecodeError:
            pass

    return None, result
