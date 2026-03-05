import json
from pathlib import Path


def build_hierarchical_law_tree(flat_data):
    """
    将扁平化法律JSON转换为层级嵌套树，节点名称包含父节点前缀，并增加摘要与向量条目
    """
    if not flat_data:
        return []

    # 提取法律名称作为根节点
    law_name = flat_data[0].get("法律", "未知法律")
    tree = {
        "名称": law_name,
        "类型": "法律",
        "摘要": "",
        "向量": [],
        "子节点": []
    }

    # 定义层级处理顺序及字段映射
    levels_config = [
        ("编", "编名"),
        ("章", "章名"),
        ("节", "节名"),
        ("条", None),
        ("款", None),
        ("项", None)
    ]

    for item in flat_data:
        current_level_list = tree["子节点"]
        parent_full_name = law_name

        for key, name_key in levels_config:
            val = item.get(key)

            # 规则：如果某一层级在原文中不存在，跳过该层
            if val == "原文未提及" or not val:
                continue

            # 构造当前节点的显示名称（如：第一章 总则）
            current_label = val
            if name_key and item.get(name_key) != "原文未提及":
                current_label = f"{val} {item.get(name_key)}"

            # 拼接父节点名称以实现全路径显示
            full_node_name = f"{parent_full_name} > {current_label}"

            # 查找是否已存在该层级节点
            existing_node = next((n for n in current_level_list if n["名称"] == full_node_name), None)

            if not existing_node:
                new_node = {
                    "名称": full_node_name,
                    "层级": key,
                    "内容": "",
                    "向量": [],
                    "子节点": []
                }

                # 判定规则：若当前记录层级匹配，则填入该节点自身的正文内容
                if item.get("层级") == key:
                    new_node["内容"] = item.get("内容", "")

                current_level_list.append(new_node)
                existing_node = new_node
            else:
                # 如果节点已存在（例如先创建了“条”节点，后续处理其下的“款”），
                # 且当前记录正是该“条”级自身的总述内容，则补充内容
                if item.get("层级") == key and not existing_node["内容"]:
                    existing_node["内容"] = item.get("内容", "")

            # 深入下一层级
            parent_full_name = full_node_name
            current_level_list = existing_node["子节点"]

    return tree


def clean_tree(node):
    """递归清理空子节点列表，保持JSON整洁"""
    if "子节点" in node:
        if not node["子节点"]:
            del node["子节点"]
        else:
            for child in node["子节点"]:
                clean_tree(child)
    return node


if __name__ == "__main__":
    # 假设输入路径为 main.py 中定义的输出位置
    input_path = "../data/json/test_output.json"
    output_path = "../data/json/2.json"

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 构建并清理树结构
        final_tree = build_hierarchical_law_tree(data)
        final_tree = clean_tree(final_tree)

        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_tree, f, ensure_ascii=False, indent=2)

        print(f"树状结构生成成功！\n文件路径: {output_path}")
        print("每个节点已包含：名称(含父路径)、层级、内容、摘要(留空)、向量(空列表)")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")