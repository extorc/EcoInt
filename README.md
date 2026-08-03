# Ecoint

## TODO
- [ ] Ingest data
  - *Explanation:* The current GraphRAG query routing system is live and fully working. However, in the current sparse state of the database, nodes might be genuinely unrelated but superficially connected through random articles. Real, high-quality logical and causal connections between entities can only emerge once a massive volume of articles is fetched, extracted, and ingested into the graph database, providing the necessary density for graph traversals to become highly accurate.
- [ ] Refine LLM Entity Extraction
  - *Explanation:* The current extraction logic often pulls overly generic terms (e.g., "Government", "PM", "Budget") or temporal words (e.g., "July", "September"). This leads to false equivalencies during semantic merging (e.g., improperly linking unrelated events just because they happened in the same month, or conflating "UK Government" with "Indian PM" under generic terms). The extraction prompt needs strict constraints to filter out generic/temporal terms and only extract highly specific, disambiguated proper nouns.
