import json
import asyncio
from pathlib import Path
import config  # 确保 config.py 在同一目录下
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class LawProcessor:
    def __init__(self):
        # 统一从 config 读取字段
        self.api_key = getattr(config, "EMBEDDING_API_KEY", None)
        self.base_url = getattr(config, "base_url", "https://api.siliconflow.cn/v1")

        # 建议在 config.py 中添加 embedding_model = "BAAI/bge-m3"
        # 如果没定义，这里设置一个 SiliconFlow 默认常用的向量模型
        self.embedding_model = getattr(config, "EMBEDDING_MODEL_NAME")

        # 初始化 Embedding
        self.embeddings = OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.embedding_model,
            dimensions=4096
        )

    async def generate_vector(self, text: str) -> list:
        """调用接口生成向量"""
        if not text.strip():
            return []
        try:
            # 执行向量化
            return await self.embeddings.aembed_query(text)
        except Exception as e:
            # 如果报错 400 Model does not exist，通常是 config 里的 embedding_model 写错了
            print(f"\n[向量错误] 模型 '{self.embedding_model}' 调用失败: {e}")
            return []

    # 取向量平均值
    async def average_vectors(self, vectors: list) -> list:
        if not vectors:
            return []
        try:
            # 计算平均向量
            avg_vector = [sum(col) / len(vectors) for col in zip(*vectors)]
            return avg_vector
        except Exception as e:
            print(f"\n[向量错误] 平均向量计算失败: {e}")
            return []

    async def process_node(self, node: dict):
        """递归处理节点：后序遍历"""
        vector_list = []

        # 获取该节点内容的向量
        if "内容" in node and node["内容"].strip():
            print(f"正在处理节点 -> {node['名称']}...\n", end="", flush=True)
            content_vector = await self.generate_vector(node["内容"])
            if content_vector:
                vector_list.append(content_vector)

        # 递归处理子节点
        if "子节点" in node and node["子节点"]:
            for child in node["子节点"]:
                await self.process_node(child)
                if child.get("向量"):
                    vector_list.append(child["向量"])

        # 计算平均向量
        if vector_list:
            node["向量"] = await self.average_vectors(vector_list)
        else:
            node["向量"] = []  # 确保没有向量时是空列表

    def run(self, input_file, output_file):
        if not Path(input_file).exists():
            print(f"错误：找不到文件 {input_file}")
            return

        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"使用模型: Embedding={self.embedding_model}")
        print("开始递归处理法律树 (从叶子节点向上)...")

        asyncio.run(self.process_node(data))

        # 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！文件已保存至: {output_file}")


if __name__ == "__main__":
    processor = LawProcessor()
    # 路径匹配你 TreeRAG 项目的 data 目录结构
    processor.run("../data/json/中华人民共和国劳动法_structured_hierarchical.json", "../data/json/3中华人民共和国劳动法_structured_hierarchical.json")