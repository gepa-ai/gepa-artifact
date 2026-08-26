Background (HotpotQA Agent)

This HotpotQA agent is a multi-hop QA pipeline (HotpotMultiHop) with four sequential modules:
1) summarize1: Summarizes the first-hop retrieved passages and outputs summary_1.
2) create_query_hop2: Generates the second-hop retrieval query from question and summary_1.
3) summarize2: Summarizes the second-hop passages conditioned on summary_1 and outputs summary_2.
4) final_answer: Produces the final answer from question, summary_1, and summary_2.

For the baseline run, metric_logs contains the training traces (train_traces.jsonl) and per-example trace files (traces/trace_XXXX.json). Each trace includes:
- question / gold_answer
- prediction (answer, hop1_docs, hop2_docs)
- per-module dialogue trace (system prompt, user inputs, assistant outputs)
- metric_output (overall answer EM, true/false)

Dataset split note: this benchmark only uses the HotpotQA train split, then applies a 40/40/20 split and trimming, resulting in train=150, val=300, test=300. The train traces in this folder come from that train subset.
