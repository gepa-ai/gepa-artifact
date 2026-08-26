Given the fields `question`, `passages`, produce the fields `summary`.

## Instructions

1. **Keep summaries concise (under 500 characters).** Focus on key facts directly relevant to answering the question. Omit background details, historical context, or explanatory text unless essential.

2. **Extract key entities, names, dates, and relationships as concise facts.** Avoid complete sentences when a phrase suffices. Format: 'Entity: fact' rather than 'The entity is described as having this fact.'

3. **Read the question carefully.** Prioritize extracting facts that directly relate to what the question asks. If the question asks about a person's role, focus on roles and occupations. If it asks about events, focus on event names and dates.
