import json
import numpy as np
import sys
from pathlib import Path
from langchain_openai import OpenAIEmbeddings

import config


class TreeSearcher:
    def __init__(
            self,
            json_path: str,
            beam_width: int = 3,  # 默认搜索广度
            api_key = config.EMBEDDING_API_KEY,
            base_url: str = config.EMBEDDING_BASE_URL,
            model_name: str = config.EMBEDDING_MODEL_NAME
    ):
        """
        初始化搜索器
        :param json_path: 向量化后的JSON文件路径
        :param beam_width: 束搜索宽度 (Beam Width)，决定DFS在每层下探多少个分支
        """
        self.json_path = json_path
        self.beam_width = beam_width
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

        # 初始化 Embedding 客户端
        self.embeddings = OpenAIEmbeddings(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_name,
            dimensions = 4096
        )

        self.data = self._load_data()

    def _load_data(self):
        """加载 JSON 数据"""
        if not Path(self.json_path).exists():
            raise FileNotFoundError(f"文件未找到: {self.json_path}")

        print(f"正在加载知识库树: {self.json_path} ...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def _cosine_similarity(self, vec1, vec2):
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def _dfs_search(self, node, query_vec, results_collector):
        """
        深度优先搜索 (DFS) + 剪枝逻辑
        """
        # 1. 终止条件：如果是叶子节点（没有子节点），或者达到了最底层（条/款）
        # 计算该节点自身的相似度并加入结果集
        if not node.get("子节点"):
            # 计算当前叶子节点的相似度
            sim = self._cosine_similarity(query_vec, node.get("向量"))
            if sim > 0:
                results_collector.append({
                    "score": sim,
                    "path": node.get("名称", "未知路径"),
                    "content": node.get("内容", "")
                })
            return

        # 2. 如果是中间节点（编、章、节），评估其所有子节点的相似度
        children_candidates = []
        for child in node["子节点"]:
            if child.get("向量"):
                sim = self._cosine_similarity(query_vec, child["向量"])
                children_candidates.append((sim, child))

        # 3. 剪枝 (Pruning)：按相似度降序排列
        children_candidates.sort(key=lambda x: x[0], reverse=True)

        # 4. 只保留 Top-N (beam_width) 个高分分支进行递归下探
        # 这一步体现了“按相似度DFS”的核心：低分分支被直接剪掉，不再遍历其内部
        top_children = children_candidates[:self.beam_width]

        for score, child_node in top_children:
            # 只有相似度 > 0 才继续（避免完全无关的分支）
            if score > 0:
                self._dfs_search(child_node, query_vec, results_collector)

    def search(self, query: str):
        """执行搜索"""
        print(f"\n正在向量化问题: '{query}' ...")
        try:
            query_vec = self.embeddings.embed_query(query)
        except Exception as e:
            print(f"向量化接口调用失败: {e}")
            return []

        print(f"开始相似度 DFS 检索 (每层保留前 {self.beam_width} 个分支)...")
        results = []

        # 兼容根节点是列表（多个法）或字典（单个法）的情况
        root_nodes = self.data if isinstance(self.data, list) else [self.data]

        # 从根部开始递归
        for root in root_nodes:
            self._dfs_search(root, query_vec, results)

        # 对最终收集到的所有叶子节点结果进行全局排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results


def get_user_input(prompt, default=None, val_type=str):
    """辅助函数：获取用户输入并处理默认值"""
    while True:
        try:
            val_str = input(f"{prompt} (默认: {default}): ").strip()
            if not val_str:
                if default is not None:
                    return default
                else:
                    continue

            # 去除引号
            val_str = val_str.replace('"', '').replace("'", "")

            if val_type == int:
                return int(val_str)
            return val_str
        except ValueError:
            print("输入格式错误，请重试。")


def main():
    print("=" * 60)
    print("      基于树状相似度DFS的法律检索系统")
    print("=" * 60)

    # 1. 获取文件路径
    json_path = get_user_input(
        "\n请输入向量化JSON文件路径",
        default="../data/json/3.json"
    )
    if not Path(json_path).exists():
        print(f"错误: 文件 {json_path} 不存在。")
        sys.exit(1)

    # 2. 获取搜索广度 (Beam Width)
    # 这决定了 DFS 在每一层（如“章”这一层）会保留几个最相似的“章”继续往下找
    beam_width = get_user_input(
        "请输入搜索广度 (Beam Width，每层保留几个分支?)",
        default=3,
        val_type=int
    )

    # 3. 初始化搜索器
    try:
        searcher = TreeSearcher(json_path=json_path, beam_width=beam_width)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 4. 交互循环
    while True:
        print("\n" + "-" * 50)
        query = input("请输入法律问题 (输入 q 退出): ").strip()

        if query.lower() in ['q', 'quit', 'exit']:
            print("再见！")
            break
        if not query:
            continue

        # 5. 获取结果显示数量 (Top K)
        top_k = get_user_input(
            "请输入要显示的匹配结果数量 (Top K)",
            default=3,
            val_type=int
        )

        # 6. 执行搜索
        results = searcher.search(query)

        # 7. 展示结果
        if not results:
            print("未找到相关条文。")
        else:
            print(f"\n成功找到 {len(results)} 个潜在相关条目，为您显示 Top {top_k}:")

            # 截取前 top_k 个结果
            display_results = results[:top_k]

            for i, item in enumerate(display_results, 1):
                print(f"\n[结果 {i}] 相似度: {item['score']:.4f}")
                print(f"路径: {item['path']}")

                content = item['content']
                # 内容过长则截断显示
                if len(content) > 200:
                    content = content[:200] + "..."
                elif not content:
                    content = "(无具体正文)"

                print(f"内容: {content}")


if __name__ == "__main__":
    main()