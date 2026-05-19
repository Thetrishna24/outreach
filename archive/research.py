"""Research Agent for the outreach tracker.

Given a name and institution (and optional hint), produces a structured
JSON profile by autonomously searching and reading web sources.

Uses Anthropic's native web_search tool plus our custom web_fetch tool.
"""

import json
from anthropic import Anthropic

from src.agents.base import Agent, Tool, AgentResult
from src.tools.web import web_fetch


RESEARCH_SYSTEM_PROMPT = """You are a research agent that builds accurate profiles of academic researchers for the purpose of personalized outreach.

Given a name and institution, your job is to find and synthesize:
- Their primary research areas (specific subfields, not generic labels like "biology" or "computer science")
- 2-4 recent papers or projects with titles and brief descriptions
- Their lab or group affiliation
- Their faculty/lab page URL
- Any notable recent talks, awards, or news from the last 24 months
- Best contact email if publicly available

Your process:
1. Start with a web search for the researcher at their institution. Try queries like "{name} {institution} faculty" or "{name} {institution} lab".
2. Fetch the most authoritative URL (university faculty page is best, then lab page, then Google Scholar). Use web_fetch to read the actual page content.
3. If recent papers aren't on the faculty page, search again with queries like "{name} {institution} recent papers" or "{name} 2024 paper".
4. Cross-reference at least two sources before claiming any specific fact.
5. If you cannot verify a claim, OMIT it rather than guessing.

Critical rules — these are non-negotiable:
- NEVER fabricate research areas, paper titles, affiliations, or contact info. If you don't have a source for it, leave it out.
- ALWAYS include the source URL for each claim in the `sources` field. Every URL listed must be one you actually fetched.
- Distinguish between "primary research area" (what they spend most time on) and "interests" (peripheral topics).
- If you find multiple researchers with the same name, identify which one matches the institution. If still ambiguous, set `confidence` to "low" and add an entry to `ambiguity_flags`.
- Specific subfields beat generic labels. "Computational cancer genomics with Hi-C integration" is good. "Bioinformatics" is too generic.

Output format: When you have enough information, respond with ONLY a JSON object matching this schema. Do not wrap it in markdown code fences. Do not include any text before or after the JSON.

{
  "name": "string",
  "institution": "string",
  "title": "string or null",
  "department": "string or null",
  "lab_or_group": "string or null",
  "primary_research_areas": ["string", ...],
  "recent_work": [
    {
      "title": "string",
      "year": "string or null",
      "summary": "string",
      "url": "string or null"
    }
  ],
  "lab_url": "string or null",
  "scholar_url": "string or null",
  "contact_email": "string or null",
  "notes": "string (1-3 sentences synthesizing what makes their work distinctive)",
  "sources": ["url", ...],
  "confidence": "high | medium | low",
  "ambiguity_flags": ["string", ...]
}

Required fields: name, institution, primary_research_areas (at least 1), notes, sources (at least 1), confidence, ambiguity_flags (can be empty array).
All other fields can be null or empty arrays if you don't have verified information.
"""


def build_research_agent(client: Anthropic | None = None) -> Agent:
    """Construct a Research Agent with the right tools and prompt."""
    tools = [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 8,
        },
        # custom web_fetch tool
        Tool(
            name="web_fetch",
            description=(
                "Fetch the full text content of a specific URL. "
                "Use this after web_search to read the most relevant pages. "
                "Returns cleaned text, truncated to 8000 characters."
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
        name="research_agent",
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        tools=tools,
        model="claude-haiku-4-5",
        max_iterations=10,
        max_tokens=4096,
        client=client,
    )


def research_contact(
    name: str,
    institution: str,
    hint: str | None = None,
    client: Anthropic | None = None,
) -> tuple[dict | None, AgentResult]:
    """Research a contact and return a structured profile.

    Returns a tuple of (profile_dict, agent_result).
    profile_dict is None if parsing failed; agent_result has full debug info.
    """
    agent = build_research_agent(client=client)

    user_input = f"Build a research profile for: {name}, {institution}"
    if hint:
        user_input += f"\nContext hint: {hint}"

    result = agent.run(user_input)

    if not result.success:
        return None, result

    # Try to parse the JSON response
    raw = result.output.strip()
    # Defensive: strip code fences if the model added them despite instructions
    if raw.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw

    try:
        profile = json.loads(raw)
    except json.JSONDecodeError:
        # Parsing failed — return None for the profile but keep the result for debugging
        return None, result

    return profile, result
