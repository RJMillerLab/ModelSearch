# Plan: Classification-Based Integration, LLM Serialization & Evaluation

## 1. Current State

### 1.1 Classification
- **Source**: `data/table_classifications.json` — maps `tableid` (int) → label
- **Labels**: 
  - Heuristic: `numerical`, `categorical`, `mixed`, `id_like`, `empty`, ...
  - Tab2Know: `Observation`, `Input`, `Other`, `Example`
- **Usage**: `tab2tab_by_type` filters search to same-classification tables only

### 1.2 Integration (Current)
| Source | Input | Tables | Classification-aware? |
|--------|-------|--------|----------------------|
| **Table Search** | `card2tab2card_results[search_type].intermediate` | `retrieved_table_filenames` or `table_to_models.keys()` | ❌ No |
| **Model Search** | `card2card` model_ids → relationship_parquet | model's tables (csv_basename) | ❌ No |

- Both integrate independently; no cross-source matching by label.

### 1.3 Serialization for LLM (Current)
- `evaluation.llm.serialize_table_for_prompt()`: shape + columns + CSV sample (max 50 rows × 10 cols)

---

## 2. Goal: Classification-Based Cross-Source Integration

### 2.1 “Corresponding Tables”
**Definition**: Tables with the **same classification label** across both search results.

Flow:

1. **Table Search** → tables with `retrieved_table_ids` → lookup `table_classifications.json` → get label per table
2. **Model Search** → tables from model_ids (relationship parquet) → need **filename → tableid** mapping (modellake.db) → lookup classification
3. **Intersect**: For each label L, collect:
   - Tables from Table Search with label L
   - Tables from Model Search with label L
4. **Integrate per label** (or union across labels): 
   - Option A: One integrated table per label, then union/merge
   - Option B: Single integrated table from union of all “corresponding” tables

### 2.2 Data Dependencies

| Need | Source |
|------|--------|
| tableid → label | `table_classifications.json` |
| filename/basename → tableid | `modellake.db` (modellake_index: tableid, filename) |
| Table Search tables | `retrieved_table_ids` + `table_id_to_filename` |
| Model Search tables | relationship_parquet (modelId → csv_basename) |

---

## 3. Proposed Integration Pipeline

```
┌─────────────────────┐     ┌──────────────────────┐
│  Table Search       │     │  Model Search        │
│  (single_column /   │     │  (card2card dense)   │
│   keyword / etc.)   │     │                      │
└─────────┬───────────┘     └──────────┬───────────┘
          │                             │
          ▼                             ▼
┌─────────────────────┐     ┌──────────────────────┐
│ retrieved_table_ids │     │ model_ids → tables   │
│ table_id_to_filename│     │ (relationship_parquet)│
└─────────┬───────────┘     └──────────┬───────────┘
          │                             │
          ▼                             ▼
┌──────────────────────────────────────────────────┐
│  classification_json (tableid → label)            │
│  modellake.db (filename → tableid)                │
└──────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────┐
│  Group tables by label L                          │
│  tables_table_search[L] ∪ tables_model_search[L]  │
└──────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────┐
│  Integrate (union/intersection) per label OR      │
│  integrate all corresponding tables together     │
└──────────────────────────────────────────────────┘
```

### 3.1 Implementation Steps

1. **Add `integrate_tables_from_both_sources_with_classification()`** in `table_integration.py`:
   - Input: search_results.json
   - Load classification JSON
   - Build filename→tableid from modellake.db
   - For Table Search: tableid → label
   - For Model Search: basename → tableid → label
   - Group tables by label
   - Integrate per label (or merge strategy) → output integrated DataFrame(s)

2. **API**: New endpoint e.g. `/api/integrate-cross-source` or extend existing integrate to accept mode `cross_source_by_classification`.

---

## 4. Serialization for LLM

### 4.1 Current
- Shape + column list + CSV sample (truncated)

### 4.2 Proposed

1. **Include classification context** in serialization:
   ```
   [Table: Table Search Integration]
   Classification: numerical
   Shape: 50 rows × 8 columns
   Columns: model_id, accuracy, params, ...
   Sample: ...
   ```

2. **Per-label integrated tables** (if integrating per label):
   - Serialize each label’s table separately
   - Structure: `{ "numerical": "<serialized>", "categorical": "<serialized>" }`

3. **Token budgeting**:
   - Keep `max_rows=50`, `max_cols=10` for evaluation
   - Optional: adaptive truncation by total token limit

4. **New helper**: `serialize_integrated_tables_with_metadata(df, source, classification=None)` in evaluation/llm.py

---

## 5. Evaluation Prompt Design

### 5.1 Current Issues
- Prompt says “diversity” but criteria focus on “quality”
- No clear handling of classification labels
- Output format fixed; may not fit all use cases

### 5.2 Proposed Evaluation Prompt Structure

```markdown
## Role
You are an expert data analyst evaluating integrated tables from a model search system.

## Context
- Original Query: {query}
- Table 1 (Table Search): {table1_serialized}
- Table 2 (Model Search): {table2_serialized}
- Classification (if available): Both tables contain tables of type(s): {labels}

## Task
Compare the two integrated tables on:

1. **Relevance**: How well do results match the query? (0-100)
2. **Data Quality**: Completeness, consistency, structure (0-100)
3. **Coverage**: Breadth of models/tables, information depth (0-100)
4. **Usefulness**: Practical value for model selection/analysis (0-100)
5. [Optional] **Diversity**: Semantic diversity within results (0-100)

## Evaluation Criteria
- Use the same scoring scale (0-100) for all dimensions
- Cite specific evidence (columns, sample values) when possible
- If classification metadata is present, note whether both sources align on table types

## Output Format (JSON)
{
  "comparison_score": {
    "table_search_quality": <0-100>,
    "model_search_quality": <0-100>,
    "overall_difference": <number>,
    "winner": "table_search" | "model_search" | "tie"
  },
  "dimension_scores": {
    "relevance": { "table_search": <0-100>, "model_search": <0-100> },
    "data_quality": { ... },
    "coverage": { ... },
    "usefulness": { ... }
  },
  "quality_analysis": { "table_search": {...}, "model_search": {...} },
  "key_differences": ["...", "..."],
  "recommendation": "...",
  "comparison_summary": "...",
  "classification_note": "..."  // optional, if labels used
}
```

### 5.3 Design Choices

| Choice | Rationale |
|--------|-----------|
| Explicit dimensions | Clear, comparable scores across runs |
| Optional diversity | Add only when diversity is a design goal |
| Classification note | Lets evaluator use label info when available |
| Evidence citation | Improves consistency and debuggability |

---

## 6. Implementation Order

| Phase | Task | Effort |
|-------|------|--------|
| 1 | Add filename→tableid mapping (modellake.db) | Small |
| 2 | Add `integrate_tables_from_both_sources_with_classification()` | Medium |
| 3 | Extend serialization to include classification + metadata | Small |
| 4 | Update evaluation prompt in `evaluation/prompt.py` | Small |
| 5 | Wire new integration into backend API + frontend | Medium |
| 6 | Add tests / manual checks on sample jobs | Small |

---

## 7. Open Questions

1. **Union vs per-label**: One big integrated table vs one per label?
2. **Fallback**: If classification JSON is missing for some tables, integrate anyway or skip?
3. **Model Search tableids**: Does relationship_parquet or modellake hold a direct filename→tableid mapping, or do we need an extra lookup?
4. **Evaluation scope**: Should evaluation always assume two integrated tables (Table vs Model), or support N tables (e.g. one per label)?

---

## 8. Files to Modify

| File | Change |
|------|--------|
| `src/integration/table_integration.py` | New cross-source integration + classification grouping |
| `src/search/classification.py` | Potentially: filename→tableid helper (or in integration) |
| `src/evaluation/llm.py` | Serialization with classification metadata |
| `src/evaluation/prompt.py` | Updated evaluation prompt |
| `src/demo/backend.py` | New/updated integrate API |
| `src/demo/frontend.py` | UI for cross-source integration mode |
