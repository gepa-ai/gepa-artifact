Given the fields `question`, `summary_1`, produce the fields `query`.

Teacher-corrected few-shot examples (from incorrect training traces):
Example 1:
Input:
Question: Howard Webb was the referee in a game that saw Bayern Munich lose by two goals by a team who had won the European cup back to back in which two years ?
Summary 1: The match in question involved Bayern Munich losing by two goals against a team that achieved back-to-back European Cup victories, which aligns with Manchester United's history (European Cup wins in 1968 and 1999). However, the provided passages do not directly reference any such game officiated by Howard Webb, nor do they provide two consecutive European Cup-winning years that match the scenario described. Based on inferences, further external context is necessary to answer the question definitively.
Output:
Reasoning: The question involves a specific football match scenario, where Howard Webb officiated a game in which Bayern Munich lost by two goals to a team that had previously won the European Cup in back-to-back years. The summary highlights that Manchester United's history includes European Cup wins in 1968 and 1999 but does not specify a match meeting the description. Furthermore, there is no direct confirmation of the team or years in question, nor explicit mention of the match officiated by Howard Webb. Thus, additional information is required on Howard Webb's officiating history and a match fitting the criteria.
Query: Search for a match officiated by Howard Webb where Bayern Munich lost by two goals to a team that had won the European Cup in consecutive years. Identify the relevant European Cup-winning team and their back-to-back victory years.
----
Example 2:
Input:
Question: If Thomas Mitchener produced Frank Carter & The Rattlesnakes second studio album, what year would Catherine Marks had to have worked with the band?
Summary 1: Catherine Marks worked with Frank Carter & The Rattlesnakes at some point, but no specific timeline or project linking her to them in 2017 or the "Modern Ruin" album (produced by Thomas Mitchener) is detailed in the passages.
Output:
Reasoning: The question seeks to determine the timeline of when Catherine Marks worked with Frank Carter & The Rattlesnakes. According to the summary, there is no confirmation that Catherine Marks worked with the band specifically on the "Modern Ruin" album, which was released in 2017 and produced by Thomas Mitchener. Without additional context establishing a specific project or timeline for Catherine Marks' involvement with the band, the year of her collaboration cannot be deduced from the given information.
Query: What specific year or project did Catherine Marks work on with Frank Carter & The Rattlesnakes?
----
