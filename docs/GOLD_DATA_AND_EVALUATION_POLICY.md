# Gold Data And Evaluation Policy

Last updated: 2026-05-26

## Policy goals

1. Keep benchmark claims tied to verified, auditable evidence.
2. Separate data-quality/alignment issues from OCR model/runtime issues.
3. Prevent overclaiming readiness from narrow or unverified slices.

## Data and alignment rules

- Use explicit benchmark manifests/splits and keep them versioned.
- Run alignment checks before promotion decisions.
- Score only safe_to_use_for_scoring records for readiness claims.
- Maintain manual-review queues for ambiguous records.

## Annotation conversion rules

- Conversion should be deterministic and traceable.
- Do not alter source gold files during conversion.
- Emit explicit warnings for malformed or missing rows.

## Evaluation reporting minimums

1. CER/WER plus failed/empty rates.
2. Runtime p95/p99 or equivalent tail indicators.
3. Per-dataset, per-document-type, and per-language/script slices where available.
4. Explicit report of excluded records and reasons.

## Readiness claim constraints

- No broad private-beta claim without broad evidence coverage.
- Keep receipts and Latn+Hang exclusions unless explicitly revalidated.
- Category-only improvements cannot be treated as global readiness.

## Evidence expansion expectations

- Track newly verified pages added per phase.
- Require non-regression on smoke and expanded sets before scoped promotion.
- Keep policy gating and mandatory review active while evidence remains uneven.

## Related docs

- docs/HOW_TO_RUN_BENCHMARKS.md
- docs/PRIVATE_BETA_READINESS_POLICY.md
- docs/GOLDSET_EVALUATION.md
