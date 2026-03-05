import json
from pathlib import Path


# 法律文本的层次化器
class Hierarchizer:
    def __init__(
            self,
            input_path: str = "../data/json/test_output.json",
            output_path: str = "../data/json/2.json"
    ):
        """
        初始化层次化器
        :param input_path: 扁平化JSON数据(Structurer输出)的输入路径
        :param output_path: 层次化树状JSON的输出路径
        """
        self.input_path = input_path
        self.output_path = output_path

        # 定义层级处理顺序及字段映射
        self.levels_config = [
            ("编", "编名"),
            ("章", "章名"),
            ("节", "节名"),
            ("条", None),
            ("款", None),
            ("项", None)
        ]

    def _build_hierarchical_tree(self, flat_data):
        """内部方法：将扁平化法律JSON转换为层级嵌套树"""
        if not flat_data:
            return {}

        # 提取法律名称作为根节点
        law_name = flat_data[0].get("法律", "未知法律")
        tree = {
            "名称": law_name,
            "类型": "法律",
            "向量": [],
            "子节点": []
        }

        for item in flat_data:
            current_level_list = tree["子节点"]
            parent_full_name = law_name

            for key, name_key in self.levels_config:
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
                    # 如果节点已存在，且当前记录正是该节点的总述内容，则补充内容
                    if item.get("层级") == key and not existing_node["内容"]:
                        existing_node["内容"] = item.get("内容", "")

                # 深入下一层级
                parent_full_name = full_node_name
                current_level_list = existing_node["子节点"]

        return tree

    def _clean_tree(self, node):
        """内部方法：递归清理空子节点列表，保持JSON整洁"""
        if "子节点" in node:
            if not node["子节点"]:
                del node["子节点"]
            else:
                for child in node["子节点"]:
                    self._clean_tree(child)
        return node

    def process(self):
        """
        执行完整层次化流程：读取文件 -> 构建树 -> 清理 -> 保存文件
        """
        try:
            # 1. 读取数据
            if not Path(self.input_path).exists():
                raise FileNotFoundError(f"输入文件不存在: {self.input_path}")

            print(f"正在读取文件: {self.input_path} ...")
            with open(self.input_path, "r", encoding="utf-8") as f:
                flat_data = json.load(f)

            # 2. 构建与清理
            print("正在构建层级结构...")
            raw_tree = self._build_hierarchical_tree(flat_data)
            final_tree = self._clean_tree(raw_tree)

            # 3. 保存结果
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(final_tree, f, ensure_ascii=False, indent=2)

            print(f"树状结构生成成功！")
            print(f"文件路径: {self.output_path}")
            print("每个节点已包含：名称(含父路径)、层级、内容、向量(空列表)")

            return final_tree

        except Exception as e:
            print(f"Hierarchizer 处理过程中出现错误: {e}")
            raise e


if __name__ == "__main__":
    # 使用示例
    input_file = "../deprecated/test_output.json"
    output_file = "../deprecated/2.json"

    hierarchizer = Hierarchizer(input_path=input_file, output_path=output_file)
    hierarchizer.process()