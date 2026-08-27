# gtfs-journey-plan-eval

Schedule-constrained journey planning: auditing zero-shot LLM itineraries against
exact transit gold.

We ask **Ox Alpha Free** (`x-preview-f-free`, served through OpenCode Zen) to plan
public-transit journeys on real GTFS feeds (Portland TriMet, Chicago CTA,
Helsinki HSL, NYC Subway) and verify every leg against the actual timetable.

Gold journeys are computed with a RAPTOR-style transit router over active GTFS
service. The benchmark uses public stop-name semantics: a displayed stop name
represents all GTFS stop IDs sharing that name, and gold optimization considers
all compatible origin/destination stop IDs.

## Design

Three evidence arms separate different sources of planning failure:

- **closed_book** — the model plans from general network knowledge alone;
- **schedule_excerpt** — the prompt injects automatically generated local timetable
  evidence, but is not guaranteed to contain the complete fastest journey;
- **oracle_complete_schedule** — the prompt contains sufficient schedule evidence
  for at least one feasible journey, selected using the hidden fastest RAPTOR
  journey, together with same-route timing distractors.

The oracle arm does **not** reveal GTFS `trip_id` values or explicitly provide the
hidden leg order. Because its evidence is selected from the hidden fastest route
and endpoints, however, the route chain is strongly informative by construction.
It should therefore be interpreted as an **evidence-sufficiency upper bound**, not
as a realistic retrieval system.

Geographic walking transfers are capped at 300 m; explicit GTFS transfer edges are
also honored. The router and evaluator use matching transfer and public-stop-name
semantics.

Scoring distinguishes:

- destination reach;
- *lenient feasibility* — a coherent route/stop/walk chain reaches the destination;
- *strict feasibility* — the chain is also consistent with the real timetable and
  temporal transitions;
- abstention / empty itineraries;
- unresolved stops, hallucinated routes, time mismatches, bad transitions, and
  invalid walks;
- stated-arrival optimality gap relative to gold.

## Experiment

Service date: **2026-08-26**

The frozen benchmark contains:

| Feed | Reachable queries |
|---|---:|
| TriMet | 192 |
| CTA | 200 |
| HSL | 189 |
| MTA | 192 |
| **Total** | **773** |

With three evidence arms, the intended experiment contains **2,319 evaluations**.

The OpenCode Zen run returned **2,243 successful scored evaluations**. The remaining
**76 evaluations were provider/network failures** during the original run
(75 upstream HTTP 5xx responses and one timeout). A later retry occurred after the
provider stopped accepting the model identifier.

Provider failures are treated as **missing data, not model failures**. No alternate
model or serving route was used to fill them.

The primary analysis uses the **710 queries with successful responses in all three
arms** (2,130 paired evaluations).

## Findings

| Arm | Destination reached | Lenient feasible | Strict feasible |
|---|---:|---:|---:|
| Closed book | 89.9% | 9.4% | **0.0%** |
| Schedule excerpt | 81.1% | 23.2% | **3.1%** |
| Oracle complete | 99.6% | 86.9% | **84.4%** |

- **Plausibility is not schedule reliability.** Closed-book responses frequently
  reach the named destination, but none of the 710 paired journeys survive strict
  timetable verification.
- **Local schedule evidence improves structure more than timing.** Lenient
  feasibility rises from 9.4% to 23.2%, while strict feasibility reaches only 3.1%.
- **Evidence sufficiency changes the regime.** Oracle-complete evidence raises strict
  feasibility to 84.4%, an **+81.3 percentage-point** paired improvement over the
  schedule excerpt (95% paired-bootstrap CI: **+78.3 to +84.1 pp**).
- **Failures change with evidence.** Time mismatches occur in 97.0% of closed-book,
  72.5% of excerpt, and 4.8% of oracle outputs on the paired subset. Oracle residual
  failures are dominated by transition chaining and remaining timing errors rather
  than route hallucination.
- **The result is not driven by one network.** Oracle strict feasibility remains
  roughly 82–89% across all four feeds.
- **Missing-data sensitivity is small.** Across the primary outcome rates, the
  largest difference between available-case and paired-complete estimates is
  approximately 0.57 percentage points.

The strongest supported conclusion is that schedule-constrained journey planning is
highly **evidence-sensitive**: plausible transit knowledge is insufficient for exact
execution, while sufficiently complete timetable evidence enables the same model to
produce schedule-consistent journeys at a much higher rate.

## Reproduce

```bash
make setup && make test

make fetch FEED=trimet   # repeat for cta, hsl, mta
make build FEED=trimet
make gold  FEED=trimet   # WALK=300 metres by default

PYTHONPATH=src .venv/bin/python -m gtfsplan.pipeline run \
  --feeds trimet,cta,hsl,mta \
  --arms closed_book,schedule_excerpt,oracle_complete_schedule \
  --workers 4
