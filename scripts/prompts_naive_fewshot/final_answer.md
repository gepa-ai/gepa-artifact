Given the fields `question`, `summary_1`, `summary_2`, produce the fields `answer`.

Naive few-shot examples (from baseline training traces):
Example 1:
Input:
Question: What singer born in London in 1988 joined The Voice as a new judge in 2015?
Summary 1: The singer born in London in 1988 who joined "The Voice" as a new judge in 2015 is Jessie J.
Summary 2: The singer born in London in 1988 who joined "The Voice" as a new judge in 2015 is Jessie J.
Output:
Reasoning: Both summaries agree that the singer born in London in 1988 who joined "The Voice" as a new judge in 2015 is Jessie J. Jessie J is a well-known singer and songwriter with her career encompassing both solo performances and television appearances, including her role as a judge on "The Voice". The consistency in the provided summaries and the factual nature of the information support this answer.
Answer: Jessie J
----
Example 2:
Input:
Question: Langston Hughes and Ann Beattie, are American, which occupation do they share?
Summary 1: Langston Hughes and Ann Beattie both share the occupation of being writers.
Summary 2: Langston Hughes and Ann Beattie are both American writers.
Output:
Reasoning: Both summaries indicate that Langston Hughes and Ann Beattie share the same occupation of being writers. This conclusion is drawn from their shared designation as "American writers," confirming their mutual profession.
Answer: Writer.
----
Example 3:
Input:
Question: This English actor who starred in the CBS drama "Eleventh Hour" appeared in this 1998 American-Australian Neo-noir science fiction film directed by Alex Proyas?
Summary 1: The English actor who starred in the CBS drama "Eleventh Hour" and appeared in the 1998 American-Australian neo-noir science fiction film "Dark City," directed by Alex Proyas, is Rufus Sewell.
Summary 2: Rufus Sewell is the English actor who starred in the CBS drama "Eleventh Hour" and appeared in the 1998 American-Australian neo-noir science fiction film "Dark City," directed by Alex Proyas.
Output:
Reasoning: Both summaries agree that Rufus Sewell is the English actor who starred in the CBS drama "Eleventh Hour" and also appeared in the 1998 American-Australian neo-noir science fiction film "Dark City," directed by Alex Proyas. Since both summaries corroborate the same information, we can confidently conclude that Rufus Sewell is the correct answer.
Answer: Rufus Sewell
----
