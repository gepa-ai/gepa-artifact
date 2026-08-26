Given the fields `question`, `passages`, produce the fields `summary`.

**Instructions:**
1. Carefully identify the target entity type requested in the question (e.g., person, place, film, date, occupation, owner).
2. When the question asks for "owner," identify the ultimate parent company or owner, not just the immediate operational entity.
3. Extract available facts from passages and make supported inferences rather than defaulting to "insufficient information."
4. When passages mention multiple candidates, verify each against the question criteria before selecting the answer.
5. If passages don't explicitly state certain attributes (like nationality), acknowledge the limitation but still extract the relevant entity when it's clearly mentioned.
6. Ensure the summary directly addresses what the question is asking for, with the most specific and accurate information available in the passages.
