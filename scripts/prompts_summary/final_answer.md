Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.

## Instructions

1. **Extract only the minimal answer required.** Remove unnecessary titles, qualifiers, and explanatory text. If the question asks 'Who', provide only the name. If it asks 'What', provide only the entity. If it asks for a location, provide only the location name without country or region unless specifically requested.

2. **Use precise terminology from the summaries.** When nationality or specific categorizations are involved, prefer the exact term provided in the source material.

3. **If summaries contain detailed explanations**, identify and extract only the core fact that directly answers the question.

4. **Error handling:** If summaries are missing or unclear, generate a best-effort answer based on available information. Never return empty output.
