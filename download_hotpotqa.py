#!/usr/bin/env python3
"""
下载 HotpotQA 数据集并保存到本地
"""

from datasets import load_dataset
import os
import json

def download_hotpotqa():
    """下载 HotpotQA fullwiki 数据集并保存到本地 hotpotQA 文件夹"""
    print("正在下载 HotpotQA 数据集...")
    
    # 下载数据集
    dataset = load_dataset("hotpot_qa", "fullwiki", trust_remote_code=True)
    
    print("\n数据集下载完成！")
    print(f"训练集样本数: {len(dataset['train'])}")
    print(f"验证集样本数: {len(dataset['validation'])}")
    print(f"测试集样本数: {len(dataset['test'])}")
    
    # 显示一个示例
    print("\n示例数据:")
    example = dataset['train'][0]
    print(f"问题: {example['question']}")
    print(f"答案: {example['answer']}")
    print(f"类型: {example['type']}")
    print(f"难度级别: {example['level']}")
    
    # 创建保存目录
    save_dir = "./hotpotQA"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n正在保存数据到 {save_dir} ...")
    
    # 保存为 JSON 格式
    for split in ['train', 'validation', 'test']:
        output_file = os.path.join(save_dir, f"{split}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([dict(example) for example in dataset[split]], f, ensure_ascii=False, indent=2)
        print(f"✓ 已保存 {split} 到 {output_file}")
    
    # 也保存为 parquet 格式（更高效）
    dataset.save_to_disk(os.path.join(save_dir, "dataset"))
    print(f"✓ 已保存完整数据集到 {os.path.join(save_dir, 'dataset')}")
    
    return dataset

if __name__ == "__main__":
    dataset = download_hotpotqa()
    print("\n✓ 数据下载并保存完成！")
