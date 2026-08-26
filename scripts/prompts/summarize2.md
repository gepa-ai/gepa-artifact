Given the fields `question`, `context`, `passages`, produce the fields `summary`.

**Instructions:**
1. Carefully read the question to identify what type of entity or information is being requested.
2. Use the context as your starting point and verify or expand it using the passages.
3. When passages contain relevant information, synthesize it with the context to form a comprehensive understanding.
4. Make supported inferences when direct evidence is partial but sufficient to reach a reasonable conclusion.
5. Extract directly stated facts even when the overall question is complex.
6. If the question involves entity identification, verify which entity type is being requested (name vs. role, specific vs. general, current vs. historical).
7. Maintain consistent confidence when evidence clearly supports a conclusion - avoid introducing uncertainty when facts are available.
8. Produce a summary that directly addresses the question with the most relevant and specific information available.
