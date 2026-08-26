Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.

## Instructions

Your task is to synthesize summary_1 and summary_2 into a final answer.

**Core Requirements:**
1. Base your answer ONLY on information in summary_1 and summary_2 - never add external knowledge
2. If both summaries agree, provide the answer directly
3. If summaries conflict, explain the conflict and provide the best answer based on available information
4. If neither summary answers the question, state this clearly

**Answer Format:**
- For "which/who/what" questions: Provide a concise, direct answer (often just a name or short phrase)
- For "when/where" questions: Provide the specific date/location
- For "why/how" questions: Provide a brief explanation
- Avoid unnecessary verbosity - match answer length to question complexity

**Reasoning Format:**
- Keep reasoning concise - explain only how you synthesized the summaries
- Do NOT repeat the full content of summary_1 and summary_2
- Focus on: agreements, conflicts, gaps, or how you resolved ambiguity
- If summaries are consistent and complete, minimal reasoning is needed

## Examples

**Good Example:**
Question: Which glacier is named after Philemon Beecher Van Trump?
Summary 1: Van Trump Glacier is named after P. B. Van Trump.
Summary 2: Van Trump Glacier is named after P. B. Van Trump, who ascended Mount Rainier in 1870.
Reasoning: Both summaries agree on the glacier name.
Answer: Van Trump Glacier

**Bad Example (avoid this):**
Question: Which glacier is named after Philemon Beecher Van Trump?
Summary 1: Van Trump Glacier is named after P. B. Van Trump.
Summary 2: Van Trump Glacier is named after P. B. Van Trump, who ascended Mount Rainier in 1870.
Reasoning: The question asks for the glacier named after Philemon Beecher Van Trump. Summary 1 states that Van Trump Glacier is named after P. B. Van Trump. Summary 2 confirms this and adds that he ascended Mount Rainier in 1870. Both summaries consistently identify the Van Trump Glacier as being named after him.
Answer: The glacier named after Philemon Beecher Van Trump is the Van Trump Glacier, which honors his historic first ascent of Mount Rainier in 1870.
