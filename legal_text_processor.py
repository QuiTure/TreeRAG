import sys
import os
import asyncio
from pathlib import Path

# 尝试导入同目录下的工具类和配置
import config
from utils.structurer import Structurer
from utils.hierarchizer import Hierarchizer
from utils.vectorizer import Vectorizer


def get_valid_file_path():
    """获取用户输入的有效文件路径"""
    while True:
        path_str = input("\n请输入目标文件的完整路径 (例如: ../data/raw/civil_code.txt): ").strip()
        # 去除可能存在的引号
        path_str = path_str.replace('"', '').replace("'", "")
        path = Path(path_str)

        if path.exists() and path.is_file():
            return path
        else:
            print(f"错误: 文件不存在或路径无效: {path_str}，请重新输入。")


def get_output_dir(input_path):
    """
    确定输出目录：
    1. 尝试寻找 input_path 父目录的同级 json 目录 (../data/raw -> ../data/json)
    2. 如果找不到，则在 input_path 同级目录下创建 output 目录
    """
    # 假设结构是 data/raw/file.txt -> data/json/
    potential_dir = input_path.parent.parent / "json"

    # 如果该目录存在，或者其父目录(data)存在，则使用它
    if potential_dir.exists() or potential_dir.parent.exists():
        potential_dir.mkdir(parents=True, exist_ok=True)
        return potential_dir

    # 否则在当前文件同级创建 output
    output_dir = input_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def main():
    print("=" * 60)
    print("      法律文本全流程处理工具 (Legal Text Processor)")
    print("=" * 60)

    # 1. 获取输入文件
    input_path = get_valid_file_path()

    # 提取文件名（不含扩展名），用于后续命名
    base_stem = input_path.stem

    # 确定输出目录
    output_dir = get_output_dir(input_path)
    print(f"\n[系统] 所有中间文件和结果将保存至: {output_dir.resolve()}")

    # 2. 选择操作步骤
    print("\n请选择要执行的操作步骤 (输入数字，逗号分隔，例如: 1,2,3):")
    print("1. [结构化] Structuring  (文本 -> _structured.json)")
    print("2. [层次化] Hierarchizing (扁平JSON -> _hierarchical.json)")
    print("3. [向量化] Vectorizing   (树状JSON -> _vectorized.json)")

    choices_str = input("\n您的选择: ").strip()

    # 解析选择
    selected_steps = []
    try:
        parts = choices_str.replace('，', ',').split(',')
        selected_steps = sorted(list(set([int(p.strip()) for p in parts if p.strip().isdigit()])))
    except Exception:
        print("输入格式错误，程序退出。")
        sys.exit(1)

    if not selected_steps:
        print("未选择任何步骤，程序退出。")
        sys.exit(0)

    # 3. 按顺序执行步骤
    # current_input_path 用于在步骤间传递文件路径
    current_input_path = str(input_path)

    # --- 步骤 1: 结构化 (Structuring) ---
    if 1 in selected_steps:
        print(f"\n" + "-" * 40)
        print(f">>> 正在执行步骤 1: 结构化 (Structuring)...")
        print(f"-" * 40)

        output_file = output_dir / f"{base_stem}_structured.json"

        try:
            structurer = Structurer(
                llm_api_key=config.LLM_API_KEY,
                llm_base_url=config.LLM_BASE_URL,
                llm_model_name=config.LLM_MODEL_NAME,
                input_path=current_input_path,
                output_path=str(output_file),
                prompt_path="prompts/prompt_chunk.txt"
            )
            structurer.process()

            # 更新下一步的输入路径
            current_input_path = str(output_file)
            print(f"[成功] 结构化文件已保存: {output_file.name}")

        except Exception as e:
            print(f"[失败] 结构化步骤出错: {e}")
            sys.exit(1)

    # --- 步骤 2: 层次化 (Hierarchizing) ---
    if 2 in selected_steps:
        print(f"\n" + "-" * 40)
        print(f">>> 正在执行步骤 2: 层次化 (Hierarchizing)...")
        print(f"-" * 40)

        # 检查输入是否为 JSON
        if not current_input_path.endswith('.json'):
            print(f"[警告] 步骤2需要JSON输入，但当前输入是: {current_input_path}")
            # 尝试查找是否存在上一步的结果文件
            potential_prev = output_dir / f"{base_stem}_structured.json"
            if potential_prev.exists():
                print(f"[提示] 检测到存在结构化文件，自动切换输入为: {potential_prev.name}")
                current_input_path = str(potential_prev)
            else:
                print("[错误] 无法继续，请先执行步骤1或提供JSON文件。")
                sys.exit(1)

        output_file = output_dir / f"{base_stem}_hierarchical.json"

        try:
            hierarchizer = Hierarchizer(
                input_path=current_input_path,
                output_path=str(output_file)
            )
            hierarchizer.process()

            # 更新下一步的输入路径
            current_input_path = str(output_file)
            print(f"[成功] 层次化文件已保存: {output_file.name}")

        except Exception as e:
            print(f"[失败] 层次化步骤出错: {e}")
            sys.exit(1)

    # --- 步骤 3: 向量化 (Vectorizing) ---
    if 3 in selected_steps:
        print(f"\n" + "-" * 40)
        print(f">>> 正在执行步骤 3: 向量化 (Vectorizing)...")
        print(f"-" * 40)

        # 检查输入
        if not current_input_path.endswith('.json'):
            # 尝试查找是否存在上一步的结果文件
            potential_prev = output_dir / f"{base_stem}_hierarchical.json"
            if potential_prev.exists():
                print(f"[提示] 检测到存在层次化文件，自动切换输入为: {potential_prev.name}")
                current_input_path = str(potential_prev)
            else:
                print(f"[错误] 步骤3需要JSON输入，且通常是层次化后的文件。当前输入: {current_input_path}")
                sys.exit(1)

        output_file = output_dir / f"{base_stem}_vectorized.json"

        try:
            vectorizer = Vectorizer(
                embedding_api_key=config.EMBEDDING_API_KEY,
                embedding_base_url=config.EMBEDDING_BASE_URL,
                embedding_model_name=config.EMBEDDING_MODEL_NAME,
                input_path=current_input_path,
                output_path=str(output_file),
                dimensions=4096
            )
            vectorizer.process()

            print(f"[成功] 向量化文件已保存: {output_file.name}")

        except Exception as e:
            print(f"[失败] 向量化步骤出错: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("所有选定任务执行完毕！")
    print(f"最终输出目录: {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()