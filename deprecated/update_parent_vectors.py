# 弃用



import json
import numpy as np
import asyncio
from pathlib import Path
import config  # 读取 config.py 中的 API 和模型配置
from langchain_openai import OpenAIEmbeddings


class VectorAggregator:
    def __init__(self):
        # 从 config 读取配置
        self.api_key = getattr(config, "api_key", None)
        self.base_url = getattr(config, "base_url", "https://api.siliconflow.cn/v1")
        self.embedding_model = getattr(config, "emb_model_name", "BAAI/bge-m3")

        # 初始化 Embedding 接口
        self.embeddings = OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.embedding_model
        )

    async def get_text_vector(self, text: str) -> np.ndarray:
        """获取文本的向量并转为 numpy 数组"""
        if not text or not text.strip():
            return None
        try:
            vec = await self.embeddings.aembed_query(text)
            return np.array(vec)
        except Exception as e:
            print(f"\n[错误] 向量生成失败: {e}")
            return None

    async def process_node_recursive(self, node: dict):
        """递归处理：后序遍历"""

        # 1. 如果没有子节点，说明是叶子节点，保持原向量不变
        if "子节点" not in node or not node["子节点"]:
            return np.array(node.get("向量", []))

        # 2. 递归获取所有子节点的向量
        child_vectors = []
        for child in node["子节点"]:
            child_vec = await self.process_node_recursive(child)
            if child_vec is not None and len(child_vec) > 0:
                child_vectors.append(child_vec)

        # 3. 生成父节点自身的原文向量
        # 优先使用摘要，无摘要则使用内容
        self_text = node.get("摘要", "") or node.get("内容", "")
        self_vec = await self.get_text_vector(self_text)

        # 4. 计算平均值：(自身向量 + 所有子节点向量之和) / 总数
        all_vecs_to_average = []
        if self_vec is not None:
            all_vecs_to_average.append(self_vec)
        all_vecs_to_average.extend(child_vectors)

        if all_vecs_to_average:
            print(f"正在更新父节点向量 -> {node['名称']} (聚合数量: {len(all_vecs_to_average)})")
            # 计算算术平均值
            mean_vector = np.mean(all_vecs_to_average, axis=0)
            node["向量"] = mean_vector.tolist()
            return mean_vector

        return self_vec

    def run(self, input_file: str, output_file: str):
        if not Path(input_file).exists():
            print(f"错误：找不到文件 {input_file}")
            return

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"开始聚合向量，使用模型: {self.embedding_model}")
        asyncio.run(self.process_node_recursive(data))

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n聚合完成！处理后的树已保存至: {output_file}")


if __name__ == "__main__":
    aggregator = VectorAggregator()
    # 读取你生成的 4096 位向量文件
    aggregator.run("data/json/law_tree_v2_4096.json", "data/json/law_tree_aggregated.json")