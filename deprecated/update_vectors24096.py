# 弃用



import json
import asyncio
from pathlib import Path
import config  # 读取 config.py 中的配置
from langchain_openai import OpenAIEmbeddings


class VectorUpdater:
    def __init__(self):
        # 从 config 读取字段，确保与之前的环境一致
        self.api_key = getattr(config, "api_key", None)
        self.base_url = getattr(config, "base_url", "https://api.siliconflow.cn/v1")
        # 使用 config 中的向量模型名称
        self.embedding_model = getattr(config, "emb_model_name", "BAAI/bge-m3")

        # 初始化 Embedding 接口
        self.embeddings = OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.embedding_model
        )

    async def generate_vector(self, text: str) -> list:
        """调用接口生成新向量"""
        if not text.strip():
            return []
        try:
            # 执行向量化
            return await self.embeddings.aembed_query(text)
        except Exception as e:
            print(f"\n[错误] 模型 '{self.embedding_model}' 生成向量失败: {e}")
            return []

    async def process_node(self, node: dict):
        """递归遍历树结构，更新每个节点的向量"""

        # 1. 优先使用摘要生成向量，如果没有摘要则使用内容
        source_text = node.get("摘要", "") or node.get("内容", "")

        if source_text.strip():
            print(f"正在更新向量 -> {node['名称']}...", end="", flush=True)
            new_vector = await self.generate_vector(source_text)

            # 替换原有向量
            node["向量"] = new_vector

            # 显示生成的向量维度以供确认
            dim = len(new_vector)
            print(f" [完成, 维度: {dim}]")
        else:
            print(f"跳过空内容节点 -> {node['名称']}")

        # 2. 递归处理子节点
        if "子节点" in node and node["子节点"]:
            for child in node["子节点"]:
                await self.process_node(child)

    def run(self, file_path: str):
        path = Path(file_path)
        if not path.exists():
            print(f"错误：找不到文件 {file_path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"正在使用模型 {self.embedding_model} 重新生成向量...")
        asyncio.run(self.process_node(data))

        # 覆盖保存原文件或保存为新文件
        output_path = path.parent / "law_tree_v2_4096.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n向量更新完成！新文件已保存至: {output_path}")


if __name__ == "__main__":
    updater = VectorUpdater()
    # 指定你之前生成的包含摘要的 JSON 文件路径
    updater.run("data/json/law_tree_with_vec.json")