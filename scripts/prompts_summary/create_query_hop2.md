Given the fields `question`, `summary_1`, produce the fields `query`.

## Instructions

1. **Generate queries that seek specific, distinguishing information.** If the question asks to compare or choose between entities, the query should explicitly ask for distinguishing characteristics. If the question asks 'who' or 'what', the query should target specific names or identifying facts, not general descriptions.

2. **Analyze what information is already known from summary_1 and what specific gap must be filled to answer the question.** The query should target that gap precisely. If summary_1 identifies an entity, the query should ask about that entity's relationship to the answer.

3. **Error handling:** Always generate a query, even if summary_1 is unclear or missing. If information is insufficient, create a general query based on the original question's topic. Never return empty output.
