import json
import numpy as np
import sys
from pathlib import Path
from typing import List

# LangChain 核心组件
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_react_agent  # <--- 关键：使用 ReAct Agent
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate

# 导入你的配置
import config


# ==========================================
# 1. 核心检索类 (TreeSearcher)
# ==========================================
class TreeSearcher:
    def __init__(self, json_path: str, beam_width: int = 3):
        self.json_path = json_path
        self.beam_width = beam_width

        # 使用 config.py 中的 Embedding 配置
        self.embeddings = OpenAIEmbeddings(
            base_url=config.EMBEDDING_BASE_URL,
            api_key=config.EMBEDDING_API_KEY,
            model=config.EMBEDDING_MODEL_NAME
        )
        self.data = self._load_data()

    def _load_data(self):
        if not Path(self.json_path).exists():
            return {}
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2: return 0.0
        v1, v2 = np.array(vec1), np.array(vec2)
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def _dfs_search(self, node, query_vec, results_collector):
        # 终止条件：叶子节点
        if not node.get("子节点"):
            sim = self._cosine_similarity(query_vec, node.get("向量"))
            # 相似度阈值 (可微调)
            if sim > 0.35:
                results_collector.append({
                    "score": sim,
                    "path": node.get("名称", "未知路径"),
                    "content": node.get("内容", "")
                })
            return

        # 中间节点
        children_candidates = []
        for child in node["子节点"]:
            if child.get("向量"):
                sim = self._cosine_similarity(query_vec, child["向量"])
                children_candidates.append((sim, child))

        # 剪枝
        children_candidates.sort(key=lambda x: x[0], reverse=True)
        top_children = children_candidates[:self.beam_width]

        for score, child_node in top_children:
            if score > 0.1:
                self._dfs_search(child_node, query_vec, results_collector)

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        if not self.data: return []
        try:
            query_vec = self.embeddings.embed_query(query)
        except Exception as e:
            print(f"向量化失败: {e}")
            return []

        results = []
        root_nodes = self.data if isinstance(self.data, list) else [self.data]
        for root in root_nodes:
            self._dfs_search(root, query_vec, results)

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]


# ==========================================
# 2. 全局初始化检索器
# ==========================================
# 请确保路径正确指向你的向量化文件 (例如: data/json/xxx_vectorized.json)
VECTOR_FILE_PATH = "data/json/中华人民共和国劳动法_vectorized.json"

global_searcher = None
if Path(VECTOR_FILE_PATH).exists():
    global_searcher = TreeSearcher(json_path=VECTOR_FILE_PATH, beam_width=3)
else:
    print(f"警告: 向量文件 {VECTOR_FILE_PATH} 不存在。")


# ==========================================
# 3. 定义 Agent 工具
# ==========================================
@tool
def search_law_database(query: str) -> str:
    """
    法律数据库检索工具。（但是现在只能检索《中华人民共和国劳动法》）
    输入用户的自然语言问题（如"试用期辞退赔偿"），返回相关的法律条文原文。
    不要输入复杂的句子，提取关键词查询效果更好。
    """
    if not global_searcher:
        return "错误：法律知识库未加载。"

    print(f"\n[Tool] 正在检索: {query}")
    results = global_searcher.search(query, top_k=3)

    if not results:
        return "未找到相关法律条文。"

    formatted_res = f"检索结果：\n"
    for i, item in enumerate(results, 1):
        formatted_res += f"【条文 {i}】 (匹配度: {item['score']:.2f})\n"
        formatted_res += f"出处: {item['path']}\n"
        formatted_res += f"内容: {item['content']}\n"
        formatted_res += "-" * 30 + "\n"

    return formatted_res


# ==========================================
# 4. 构建 ReAct Agent (适配 MiniMax)
# ==========================================
def main():
    print("=" * 60)
    print("      法律智能 Agent (ReAct 模式 - 适配 MiniMax)")
    print("=" * 60)

    if not global_searcher:
        print("错误：无法初始化检索器。")
        return

    # 1. 初始化模型 (使用 config.py 中的配置)
    llm = ChatOpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY,
        model=config.LLM_MODEL_NAME,
        temperature=0.1,  # 低温度有助于保持 ReAct 格式稳定
        max_tokens=4096
    )

    # 2. 绑定工具
    tools = [search_law_database]

    # 3. 定义 ReAct 提示词模板
    # 这个模板强制模型按照 Thought -> Action -> Observation 的步骤思考
    template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

    prompt = PromptTemplate.from_template(template)

    # 4. 创建 ReAct Agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 开启 verbose 可以看到思考过程
        handle_parsing_errors=True  # 容错处理：如果模型输出格式微小错误，自动修正
    )

    # 6. 交互循环
    while True:
        user_input = input("\n请输入问题 (q 退出): ").strip()
        if user_input.lower() in ['q', 'exit']: break
        if not user_input: continue

        try:
            print(">> Agent 正在思考...")
            result = agent_executor.invoke({"input": user_input})
            print(f"\n[回答]:\n{result['output']}")
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == "__main__":
    main()