Given the fields `question`, `context`, `passages`, produce the fields `summary`.

## Instructions

1. **Keep summaries concise (under 500 characters).** Extract only facts needed to answer the question. Prioritize specific names, dates, and key relationships.

2. **Use the most specific terminology available in the passages.** When multiple terms apply (e.g., 'writer' vs 'novelist', 'performer' vs 'singer'), prefer the more specific one. Preserve full proper nouns and formal names.

3. **When passages provide full names or formal titles, include them completely in the summary.** Do not abbreviate or shorten proper names.

4. **When the question asks about a specific entity (person, place, thing)**, verify that the extracted answer matches what the question is asking for. Check if there are multiple entities mentioned and select the one that fits the question criteria.

5. **Error handling:** If passages are missing, unclear, or insufficient, extract whatever information is available from the context. Never return empty output. If no passages are provided, summarize relevant facts from the context field.
