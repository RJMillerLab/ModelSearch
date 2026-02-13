# Card2Tab2Card Auto-Expand & Show 10+More – Bug Note

## Task (reverted)
- Auto-expand first joinable (single_column/multi_column) in Card2Tab2Card
- Auto-expand first model with tables under that joinable
- Default show 10 models, rest in "Show N more"

## Issue
When these changes were applied, **Start Search stopped working** on the server. Root cause unclear; possible factors:
1. **Template literal nesting** – Deep JS template nesting inside Python f-string may produce invalid JS or wrong escaping.
2. **IIFE scope** – The `(() => { ... })()` pattern with `displayCount`, `visibleModels`, `moreModels` inside the template might break in some environments.
3. **Escaping** – `modelTables.map(table => ...)` with nested template strings; table paths with quotes/special chars could break `copyTablePath('${esc}', this)`.

## What works (kept)
- `card2tab2card_results || {}` guard – Prevents crash when loading old saved searches without this field.

## How to re-implement safely
1. Implement in a separate JS file instead of inline Python template.
2. Or simplify: only add "Show N more" (like Card2Card), avoid auto-expand.
3. Test on the actual server (chippie) before pushing.
