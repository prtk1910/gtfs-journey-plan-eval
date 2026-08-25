"""Render natural-language journey queries from gold items."""

from __future__ import annotations

import hashlib
import random


PARAPHRASES = [
    (
        "How do I get from {o} to {d}? I want to leave around {time}.",
        "Plan me a public transit journey from {o} to {d}, departing near {time}.",
    ),
    (
        "I need to travel by public transport from {o} to {d}, "
        "leaving at approximately {time}. What is the best route?",
        "Using this city's transit network, what route should I take "
        "from {o} to reach {d} if I start around {time}?",
    ),
]


INSTRUCTIONS = (
    "You are a trip planner with knowledge of the {city} public transit network. "
    "Respond with ONLY a JSON object (no prose, no markdown) of the form:\n"
    '{{"legs": [{{"type": "ride"|"walk", '
    '"route": "<route name/number as shown to riders>", '
    '"from": "<stop name>", '
    '"to": "<stop name>", '
    '"depart": "HH:MM", '
    '"arrive": "HH:MM"}}]}}\n'
    'Legs must be in travel order. Use "ride" for vehicles '
    '(include the route) and "walk" for walking transfers. '
    "Times use 24-hour HH:MM. "
    'Set "verified": true only for legs whose exact times you can '
    "confirm from any provided schedule information; use false "
    "for legs planned from general knowledge. "
    "Always provide your best complete itinerary; "
    'if you believe no reasonable transit journey exists, '
    'respond {{"legs": []}}.'
)


CITY = {
    "trimet": "Portland, Oregon",
    "cta": "Chicago",
    "hsl": "Helsinki region",
    "mta": "New York City",
}


def render_query(
    item: dict,
    variant: int,
) -> str:
    """
    Render a deterministic natural-language query for one gold item.

    Python's built-in hash() is intentionally not used here because it is
    salted independently for different interpreter processes. A SHA-256
    derived seed makes paraphrase selection reproducible across runs.
    """
    seed_material = (
        item["feed"],
        item["origin"],
        item["destination"],
        item["dep_time"],
        variant,
    )

    digest = hashlib.sha256(
        repr(seed_material).encode("utf-8")
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        "big",
    )

    rng = random.Random(seed)

    tmpl = PARAPHRASES[
        variant % len(PARAPHRASES)
    ][rng.randrange(2)]

    hours, remainder = divmod(
        item["dep_time"],
        3600,
    )

    minutes = remainder // 60

    time_str = f"{hours:02d}:{minutes:02d}"

    return tmpl.format(
        o=item["origin_name"],
        d=item["destination_name"],
        time=time_str,
    )


def build_messages(
    item: dict,
    arm: str,
    excerpt: str | None,
    variant: int = 0,
) -> list[dict]:
    q = render_query(
        item,
        variant,
    )

    sys_text = INSTRUCTIONS.format(
        city=CITY.get(
            item["feed"],
            "the",
        )
    )

    content = q

    if arm in (
        "schedule_excerpt",
        "oracle_complete_schedule",
    ):
        content = (
            f"{q}\n\n"
            "Relevant schedule information for the network:\n\n"
            f"{excerpt}"
        )

    elif arm == "neighborhood":
        content = (
            f"{q}\n\n"
            "Broader neighborhood schedule information:\n\n"
            f"{excerpt}"
        )

    sys_text += (
        " IMPORTANT: keep your reasoning very brief "
        "(at most a few short steps); "
        "commit to your best answer quickly."
    )

    return [
        {
            "role": "system",
            "content": sys_text,
        },
        {
            "role": "user",
            "content": content,
        },
    ]
