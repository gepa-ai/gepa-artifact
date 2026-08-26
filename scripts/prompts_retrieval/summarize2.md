Given the fields `question`, `context`, `passages`, produce the fields `summary`.

## Instructions

Your task is to synthesize the context with information from passages to produce a summary that answers the question.

**Core Requirements:**
1. Evaluate whether passages are relevant to the question and context
2. Verify that passages support the context - if they contradict or don't support it, note this
3. Use only information explicitly in the context and passages - no speculation
4. If passages are irrelevant, acknowledge this rather than forcing them into the reasoning

**Summary Format:**
- Keep summaries concise and focused on answering the question
- One to two sentences is often sufficient
- Synthesize context + passages, don't just repeat the context
- Prioritize information that directly answers the question

**Handling Different Passage Types:**
- **Supporting passages**: Integrate with context to strengthen the answer
- **Contradictory passages**: Note the conflict and resolve it based on evidence quality
- **Irrelevant passages**: State "The retrieved passages do not provide relevant information" and base summary on context alone
- **Partial passages**: Use what's relevant and note what's missing

## Examples

**Good Example (supporting passages):**
Question: Who played Philip Marlow in The Singing Detective?
Context: Michael Gambon played Philip Marlow in The Singing Detective.
Passages: [1] Michael Gambon starred in The Singing Detective. [2] Gambon also played Dumbledore in Harry Potter films.
Reasoning: Passage [1] confirms the context. The character name and connection are consistent.
Summary: Michael Gambon played Philip Marlow in The Singing Detective.

**Good Example (irrelevant passages):**
Question: Which NFL player is younger, Billy Truax or Lance Rentzel?
Context: Lance Rentzel is younger than Billy Truax.
Passages: [1] "Query theory is a theory that proposes..." [2] "A prig is a person who shows..."
Reasoning: The passages are completely irrelevant to the question about NFL players. The context provides the answer.
Summary: Lance Rentzel is younger than Billy Truax.

**Bad Example (avoiding this):**
Question: Which NFL player is younger?
Context: Lance Rentzel is younger than Billy Truax.
Passages: [Irrelevant passages about query theory]
Reasoning: The question asks which player is younger. The context states Lance Rentzel is younger. Although the passages discuss query theory and other unrelated topics, we can apply query theory concepts to understand that the question is asking us to compare ages... ❌ (Forcing irrelevant passages into reasoning)
