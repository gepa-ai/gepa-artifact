Given the fields `question`, `summary_1`, produce the fields `query`.

Naive few-shot examples (from baseline training traces):
Example 1:
Input:
Question: What singer born in London in 1988 joined The Voice as a new judge in 2015?
Summary 1: The singer born in London in 1988 who joined "The Voice" as a new judge in 2015 is Jessie J.
Output:
Reasoning: To identify the singer born in London in 1988 who joined "The Voice" as a new judge in 2015, I utilized the information provided in the summary. The summary explicitly states that Jessie J fulfills these criteria.
Query: Who is Jessie J?
----
Example 2:
Input:
Question: Langston Hughes and Ann Beattie, are American, which occupation do they share?
Summary 1: Langston Hughes and Ann Beattie both share the occupation of being writers.
Output:
Reasoning: Langston Hughes and Ann Beattie are both known for their contributions to literature. Langston Hughes is recognized as a prominent poet, novelist, and playwright associated with the Harlem Renaissance, while Ann Beattie is acclaimed for her short stories and novels. Their shared occupation as writers directly answers the question about what they have in common professionally.
Query: What is the shared profession or occupation between Langston Hughes and Ann Beattie?
----
Example 3:
Input:
Question: This English actor who starred in the CBS drama "Eleventh Hour" appeared in this 1998 American-Australian Neo-noir science fiction film directed by Alex Proyas?
Summary 1: The English actor who starred in the CBS drama "Eleventh Hour" and appeared in the 1998 American-Australian neo-noir science fiction film "Dark City," directed by Alex Proyas, is Rufus Sewell.
Output:
Reasoning: To identify the English actor, the two key pieces of information to connect are the CBS drama "Eleventh Hour" and the 1998 American-Australian neo-noir science fiction film "Dark City," directed by Alex Proyas. Rufus Sewell is known as the lead actor in "Eleventh Hour" and also starred as the protagonist, John Murdoch, in "Dark City." This provides the answer to the question.
Query: Who is Rufus Sewell, the English actor who starred in "Dark City" and "Eleventh Hour"?
----
