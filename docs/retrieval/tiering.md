# Retrieval Tiering

## Tiers
- **FAST**: lexical + dense (optional) retrieval.
- **FUSION**: reciprocal-rank fusion over candidate lists.
- **RERANK**: optional reranker over top-N fused candidates.

## Planner
- FAST is never skipped.
- Rerank may be skipped when budgets are low or historical help-rate is below threshold.
- Decisions are recorded in `tier_plan_decision` with reasons.

## Tier stats
- Help-rate tracked per query class and tier.
- A tier is "helpful" when it introduces at least one final cited span not present before it ran.

## Persistence
- Each tier writes `retrieval_hit` rows with tier-specific scores.
- Citable flag is computed by span existence, bbox presence, and media readability.

