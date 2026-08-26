Given the fields `question`, `summary_1`, produce the fields `query`.

**Instructions:**
- Identify what NEW information is needed to answer the question that is NOT already provided in `summary_1`
- Generate a query to retrieve this missing information, not to confirm facts already established
- If `summary_1` has already answered part of the question, focus your query on the remaining unknowns
- Avoid generating queries that would simply retrieve more details about entities already identified in `summary_1`
- The query should seek information that will advance toward the final answer, not reconfirm what is already known
