"""
不需要 ColBERT 服务的 HotpotQA 简化程序
使用数据集自带的 context，模拟检索过程
"""

import dspy
from gepa_artifact.benchmarks import dspy_program

class HotpotMultiHopNoRetrieval(dspy_program.LangProBeDSPyMetaProgram, dspy.Module):
    """不需要检索服务的 HotpotQA 程序，使用数据集自带的上下文"""
    
    def __init__(self):
        super().__init__()
        self.k = 7
        self.create_query_hop2 = dspy.ChainOfThought("question,summary_1->query")
        self.final_answer = dspy.ChainOfThought("question,summary_1,summary_2->answer")
        self.summarize1 = dspy.ChainOfThought("question,passages->summary")
        self.summarize2 = dspy.ChainOfThought("question,context,passages->summary")
    
    def get_context_from_example(self, example):
        """从数据集 example 中提取上下文文档"""
        # HotpotQA 数据集格式：context 包含 titles 和 sentences
        titles = example.context['title']
        sentences = example.context['sentences']
        
        passages = []
        for title, sents in zip(titles, sentences):
            # 将每个文档格式化为 "标题 | 内容"
            content = " ".join(sents)
            passages.append(f"{title} | {content}")
        
        return passages[:self.k]  # 只返回前 k 个
    
    def forward(self, question, context=None):
        """
        question: 问题文本
        context: 包含 {'title': [...], 'sentences': [...]} 的字典
        """
        # HOP 1: 使用数据集中的上下文代替检索
        if context is None:
            raise ValueError("必须提供 context 参数（从数据集 example 中获取）")
        
        hop1_docs = self.get_context_from_example(
            dspy.Example(context=context)
        )
        
        summary_1 = self.summarize1(
            question=question, passages=hop1_docs
        ).summary

        # HOP 2: 生成查询（但仍然使用相同的文档）
        hop2_query = self.create_query_hop2(
            question=question, summary_1=summary_1
        ).query
        
        # 在没有检索服务时，我们重用相同的文档
        hop2_docs = hop1_docs  # 简化：使用相同的文档
        
        summary_2 = self.summarize2(
            question=question, context=summary_1, passages=hop2_docs
        ).summary

        # HOP 3: 生成最终答案
        hop3_answer = self.final_answer(
            question=question, summary_1=summary_1, summary_2=summary_2
        ).answer

        return dspy.Prediction(
            answer=hop3_answer,
            hop1_docs=hop1_docs,
            hop2_query=hop2_query,
            hop2_docs=hop2_docs,
        )
