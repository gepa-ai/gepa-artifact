Given the fields `question`, `passages`, produce the fields `summary`.

## Instructions

Your task is to extract information from the passages that answers the question, then produce a concise summary.

**Core Requirements:**
1. Use ONLY information explicitly stated in the passages - no speculation, inference, or interpretation
2. If passages don't contain the answer, state "The passages do not contain information to answer this question"
3. Cite which passage(s) support your summary (e.g., "According to passage [1]...")
4. Focus on information directly relevant to answering the question - avoid tangential details

**Summary Format:**
- Keep summaries concise and focused on answering the question
- Prioritize the most essential facts
- One to two sentences is often sufficient
- Avoid providing comprehensive background unless the question requires it

**What to Avoid:**
- Speculation about what "might be" or "could have" happened
- Information not explicitly in the passages
- Overly detailed summaries when a simple statement suffices
- Including passage content that doesn't help answer the question

## Examples

**Good Example:**
Question: Who was the first to ascend Mount Rainier?
Passages: [1] P. B. Van Trump and Hazard Stevens made the first documented climb of Mount Rainier in 1870. [2] Mount Rainier is the highest peak in Washington.
Reasoning: Passage [1] directly states who made the first documented climb.
Summary: P. B. Van Trump and Hazard Stevens made the first documented climb of Mount Rainier in 1870.

**Bad Example (avoid speculation):**
Question: Did Catherine Marks work on the 2017 album?
Passages: [1] Catherine Marks has worked with Frank Carter & The Rattlesnakes. [2] Thomas Mitchener produced their 2017 album Modern Ruin.
Reasoning: Catherine Marks worked with the band, but no passage mentions the 2017 album specifically.
Summary: Catherine Marks worked with Frank Carter & The Rattlesnakes at some point, possibly on the 2017 album or other projects. ❌ (Too speculative)

**Better:**
Summary: Catherine Marks has worked with Frank Carter & The Rattlesnakes, but the passages do not specify which projects or years. Thomas Mitchener produced their 2017 album Modern Ruin.
