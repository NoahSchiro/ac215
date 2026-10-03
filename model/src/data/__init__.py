"""Data ingestion + preprocessing pipeline (PLAN.md Steps 0-8).

`schema` (Step 0) is the canonical per-sample contract. Dataset-specific
file-format knowledge is confined to the Step 1 adapters; every step from
Step 2 onward must consume only `schema.Sample` records.
"""
