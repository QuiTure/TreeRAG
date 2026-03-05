import json
import asyncio
from pathlib import Path
from langchain_openai import OpenAIEmbeddings


# 法律文本的向量化器
class Vectorizer:
    def __init__(
            self,
            embedding_api_key,
            embedding_base_url: str,
            embedding_model_name: str,
            input_path: str,
            output_path: str,
            dimensions: int = 4096
    ):
        """
        初始化向量化器
        :param embedding_api_key: 向量模型 API 密钥
        :param embedding_base_url: 向量模型 API 基础 URL
        :param embedding_model_name: 向量模型名称
        :param input_path: 输入 JSON 文件路径 (通常是 Hierarchizer 的输出)
        :param output_path: 输出 JSON 文件路径
        :param dimensions: 向量维度 (默认 4096)
        """
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.embedding_model_name = embedding_model_name
        self.input_path = input_path
        self.output_path = output_path
        self.dimensions = dimensions

        # 初始化 Embedding 客户端
        self.embeddings = OpenAIEmbeddings(
            base_url=self.embedding_base_url,
            api_key=self.embedding_api_key,
            model=self.embedding_model_name,
            dimensions=self.dimensions
        )

    async def _generate_vector(self, text: str) -> list:
        """内部异步方法：调用接口生成向量"""
        if not text or not text.strip():
            return []
        try:
            return await self.embeddings.aembed_query(text)
        except Exception as e:
            print(f"\n[向量错误] 模型 '{self.embedding_model_name}' 调用失败: {e}")
            return []

    def _average_vectors(self, vectors: list) -> list:
        """内部方法：算向量列表的平均值"""
        if not vectors:
            return []
        try:
            # 按列求平均: zip(*vectors) 将多个列表的第 i 位组合在一起
            avg_vector = [sum(col) / len(vectors) for col in zip(*vectors)]
            return avg_vector
        except Exception as e:
            print(f"\n[向量错误] 平均向量计算失败: {e}")
            return []

    async def _process_node(self, node: dict):
        """
        内部异步方法：递归处理节点（后序遍历）
        逻辑：
        1. 先递归处理所有子节点，计算出子节点的向量。
        2. 计算当前节点自身内容的向量。
        3. 当前节点的最终向量 = Average(自身内容向量 + 所有子节点的向量)
        """
        child_vectors = []

        # 1. 递归处理子节点
        if "子节点" in node and node["子节点"]:
            for child in node["子节点"]:
                await self._process_node(child)
                # 收集子节点计算完成后的向量
                if child.get("向量") and len(child["向量"]) > 0:
                    child_vectors.append(child["向量"])

        # 2. 获取当前节点自身内容的向量
        content_vector = []
        node_content = node.get("内容", "").strip()

        if node_content:
            # 简单的进度提示
            node_name = node.get('名称', '未知')
            # 截取较长的名称以便显示
            display_name = (node_name[:25] + '..') if len(node_name) > 25 else node_name
            print(f"\r处理节点: {display_name:<30}", end="", flush=True)

            vec = await self._generate_vector(node_content)
            if vec:
                content_vector = vec

        # 3. 聚合逻辑：(当前内容向量 + 所有子节点向量) 的平均值
        all_vectors = []
        if content_vector:
            all_vectors.append(content_vector)

        if child_vectors:
            all_vectors.extend(child_vectors)

        # 计算并赋值
        if all_vectors:
            node["向量"] = self._average_vectors(all_vectors)
        else:
            node["向量"] = []

    async def _execute_async_process(self):
        """异步执行逻辑主体"""
        # 1. 读取数据
        if not Path(self.input_path).exists():
            raise FileNotFoundError(f"输入文件不存在: {self.input_path}")

        print(f"正在读取文件: {self.input_path} ...")
        with open(self.input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"开始向量化处理 (Model: {self.embedding_model_name})...")

        # 2. 启动递归处理
        # 兼容处理：如果根是字典（单个法律）或列表（多个法律）
        if isinstance(data, dict):
            await self._process_node(data)
        elif isinstance(data, list):
            for item in data:
                await self._process_node(item)

        print("\n\n向量化计算完成。")

        # 3. 保存结果
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"文件已保存至: {self.output_path}")
        return data

    def process(self):
        """
        同步入口方法：运行异步向量化流程
        """
        try:
            return asyncio.run(self._execute_async_process())
        except KeyboardInterrupt:
            print("\n用户中断处理。")
        except Exception as e:
            print(f"Vectorizer 处理过程中出现错误: {e}")
            raise e


if __name__ == "__main__":
    # 使用示例：模拟从 config.py 读取配置
    import config

    vectorizer = Vectorizer(
        embedding_api_key=config.EMBEDDING_API_KEY,
        embedding_base_url=config.EMBEDDING_BASE_URL,
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        input_path="../deprecated/2.json",  # 层次化后的文件
        output_path="../deprecated/3.json",  # 最终带向量的文件
        dimensions=1024
    )

    # 执行处理
    vectorizer.process()