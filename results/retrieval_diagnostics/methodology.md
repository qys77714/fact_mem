
## Methodology and Attribution Rules

### Gold/Confounder Identification
- Gold memory texts from `data/preprocessed/longmemeval_s_hybrid_golden.json`
- Matched to ingest memory IDs by EXACT text match for add_all, evermemos, RD
- For mem0 (which rewrites memory texts): exact match first, then fuzzy match
  (difflib.SequenceMatcher ratio ≥ 0.65) as fallback
- Confounders identified the same way (exact match only for non-mem0 methods)

### Token Counting
- Tokenizer: Qwen3-8B (same as experiment)
- Memory context tokens: sum of individual memory unit block tokens in formatted prompt
- Confounder tokens: tokens from memory units matched to confounder IDs
- Mixed tokens (RD): tokens from fused entries containing both gold and confounder sources
- No special tokens, system prompt, or question text included in memory token count

### RD Fusion Provenance
- Fused entries identified by `metadata.answer_fused=True`
- Source membership tracked via `metadata.fused_member_ids`
- Entry classified as "mixed" if fused_member_ids contain both gold and confounder IDs
- Mixed token count reported separately

### Truncation
- 256-token hard head-truncation on memory context block
- Verified: 0 questions exceed 256 tokens in final context
- Entry count = entries surviving in final context (not all retrieved top-50)

### Single Run Note
- This analysis uses single experimental runs (not 3-run mean)
- Add-all verified as effectively identical between models (464/470 identical)
- Process is deterministic (FAISS exact search, fixed seed) except for minor
  float precision differences in 3/470 questions
