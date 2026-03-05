import json
import numpy as np
import asyncio
import config
from typing import List, Dict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# --- 1. 向量相似度算法 ---
def cosine_similarity(v1, v2):
    if not v1 or not v2: return 0.0
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# --- 2. 增强型检索器 ---
class TreeRetriever:
    def __init__(self, tree_path: str):
        with open(tree_path, "r", encoding="utf-8") as f:
            self.tree = json.load(f)
        self.embeddings = OpenAIEmbeddings(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.emb_model_name,
            dimensions=4096
        )

    async def search(self, query: str, top_k: int = 1) -> List[Dict]:
        query_vec = await self.embeddings.aembed_query(query)
        results = []

        async def _traverse(node):
            if "子节点" not in node or not node["子节点"]:
                results.append({
                    "路径": node["名称"],
                    "内容": node.get("内容", "")
                })
                return

            scored_children = []
            for child in node["子节点"]:
                score = cosine_similarity(query_vec, child.get("向量", []))
                scored_children.append((score, child))

            scored_children.sort(key=lambda x: x[0], reverse=True)
            for i in range(min(top_k, len(scored_children))):
                await _traverse(scored_children[i][1])

        await _traverse(self.tree)
        return results


retriever = TreeRetriever("../data/json/中华人民共和国劳动法_vectorized.json")

if __name__ == "__main__":
    query = "张某于 2020 年 6 月入职某快递公司，双方订立的劳动合同约定试用期为 3 个月，试用期月工资为 8000 元，工作时间执行公司规章制度相关规定。该快递公司规章制度规定，工作时间为早 9 时至晚 9 时，每周工作 6 天（每日工作 12 小时，每周工作 72 小时）。2 个月后，张某以工作时间严重超过法律规定上限为由拒绝超时加班安排，快递公司即以张某在试用期间被证明不符合录用条件为由与其解除劳动合同。张某向劳动人事争议仲裁委员会申请仲裁，请求裁决快递公司支付违法解除劳动合同赔偿金 8000 元？"
    results = asyncio.run(retriever.search(query, top_k = 1))
    print(json.dumps(results, ensure_ascii=False, indent=2))