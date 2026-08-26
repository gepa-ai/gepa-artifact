Given the fields `question`, `summary_1`, produce the fields `query`.

## Instructions

Your task is to create a search query that retrieves additional information needed to answer the question. Summary_1 provides partial information from the first search hop; your query should seek complementary information for the second hop.

**Core Requirements:**
1. Always produce a valid search query - never output "No query needed" or similar
2. Identify what information is MISSING from summary_1 that's needed to answer the question
3. Create concise, search-optimized queries (2-6 words) rather than verbose questions
4. Focus on seeking new information, not confirming what summary_1 already states

**Query Format:**
- Use keyword-style queries that would work well in a search engine
- Good: "Kai Ken dog breed characteristics"
- Bad: "What are the origins, history, and detailed characteristics of the Kai Ken dog breed?"
- Focus on the missing information gap, not the entire question

**Strategy:**
1. Analyze what summary_1 already provides
2. Identify what's still needed to fully answer the question
3. Create a query targeting that missing information
4. Even if summary_1 seems complete, generate a complementary query to verify or expand

## Examples

**Good Example:**
Question: Which film did Saïd Taghmaoui star in that was directed by Mathieu Kassovitz in 1995?
Summary 1: Saïd Taghmaoui starred in La Haine (1995) directed by Mathieu Kassovitz.
Reasoning: Summary 1 provides the film name. A second query should verify details about this film.
Query: La Haine 1995 Mathieu Kassovitz

**Good Example:**
Question: What occupation do Franz Viehböck and Ulrich Walter share?
Summary 1: Franz Viehböck is an Austrian astronaut who visited Mir in 1991.
Reasoning: Summary 1 covers Franz Viehböck. The query should find information about Ulrich Walter.
Query: Ulrich Walter occupation career

**Bad Example (avoid this):**
Question: Which NFL player is younger, Billy Truax or Lance Rentzel?
Summary 1: Lance Rentzel is younger than Billy Truax.
Reasoning: The question is fully answered by summary_1.
Query: No further query is needed. ❌ (Never output non-queries)

**Better:**
Query: Billy Truax Lance Rentzel birth dates
