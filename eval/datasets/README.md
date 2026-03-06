# Evaluation Datasets

## Ground-truth queries (`queries.jsonl`)

Each line is a JSON object:
```json
{"query": "...", "relevant_doc_ids": ["doc_id_1", "doc_id_2"]}
```

- `query`: The user question
- `relevant_doc_ids`: List of document citation IDs that contain the answer

**How to expand:** After running the pipeline, check which citations appear and manually verify correctness. Add verified query-citation pairs.

## Out-of-scope queries (`oos_queries.jsonl`)

```json
{"query": "...", "expected": "refuse"}
```

- `query`: Out-of-scope question (not about agriculture)
- `expected`: Should always be `"refuse"` for OOS set

## Results

Eval scripts output to `eval/results/` as JSON files.
