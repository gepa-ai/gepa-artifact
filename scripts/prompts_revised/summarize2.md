Given the fields `question`, `context`, `passages`, produce the fields `summary`.

**Instructions:**
- Extract and synthesize relevant information from the context and passages to answer the question
- When the question specifies multiple attributes or constraints (e.g., "keyboardist/guitarist," "Italian-American journalist"), verify that your answer matches ALL specified attributes
- Do not assume the first matching entity is correct; check all candidates against all question requirements
- When passages provide both full and abbreviated name forms (e.g., "George Emil Bria" and "George E. Bria"), use the complete full name in the summary
- Carefully distinguish between similar entities mentioned in the passages and select the one that matches all question criteria
