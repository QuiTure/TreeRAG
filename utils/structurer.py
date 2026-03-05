import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# 法律文本的结构化器
class Structurer:
    def __init__(
            self,
            llm_api_key,
            input_path: str,
            output_path: str,
            llm_base_url: str = "https://api.siliconflow.cn/v1",
            llm_model_name: str = "Pro/deepseek-ai/DeepSeek-V3.2",
            prompt_path: str = "../prompts/prompt_chunk.txt"
    ):
        """
        初始化结构化器
        :param llm_api_key: LLM API 密钥
        :param input_path: 原始法律文本文件路径 (.txt)
        :param output_path: 结构化后 JSON 输出路径
        :param llm_base_url: LLM API 基础 URL
        :param llm_model_name: 模型名称
        :param prompt_path: 提示词模板文件路径
        """
        self.input_path = input_path
        self.output_path = output_path
        self.prompt_path = prompt_path

        # 初始化大语言模型
        self.model = ChatOpenAI(
            base_url=llm_base_url,
            api_key=llm_api_key,
            model=llm_model_name
        )

        # 加载提示词模板
        if not Path(self.prompt_path).exists():
            raise FileNotFoundError(f"提示词文件不存在: {self.prompt_path}")
        prompt_text = Path(self.prompt_path).read_text(encoding="utf-8")
        self.prompt = ChatPromptTemplate.from_template(prompt_text)

    def _call_llm(self, law_context: str):
        """内部方法：调用 LLM 进行处理"""
        messages = self.prompt.format_messages(context=law_context)

        full_content = ""
        print("Model processing start >> ", flush=True)

        try:
            for chunk in self.model.stream(messages):
                chunk_text = chunk.content if chunk.content else ""
                print(chunk_text, end="", flush=True)
                full_content += chunk_text
        except Exception as e:
            print(f"\nLLM 调用失败: {e}")
            raise e

        print("\n<< Model processing end", flush=True)
        return full_content

    def _clean_json_content(self, raw_content: str):
        """内部方法：清洗 LLM 返回的 Markdown 代码块"""
        content = raw_content.strip()
        if content.startswith("```json"):
            content = content[7:].strip()
        elif content.startswith("```"):
            content = content[3:].strip()

        if content.endswith("```"):
            content = content[:-3].strip()

        return content

    def process(self):
        """
        执行完整结构化流程：读取文本 -> 调用LLM -> 清洗JSON -> 保存文件
        """
        try:
            # 1. 读取原始文本
            if not Path(self.input_path).exists():
                raise FileNotFoundError(f"输入文件不存在: {self.input_path}")

            print(f"正在读取文件: {self.input_path} ...")
            law_context = Path(self.input_path).read_text(encoding="utf-8")

            # 2. 调用模型进行结构化
            raw_response = self._call_llm(law_context)

            # 3. 解析与清洗
            json_str = self._clean_json_content(raw_response)
            structured_data = json.loads(json_str)

            # 4. 保存结果
            Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)

            print(f"结构化完成！\n文件路径: {self.output_path}")
            return structured_data

        except json.JSONDecodeError as e:
            print(f"JSON 解析失败，请检查模型输出是否完整: {e}")
            # 保存原始文本以便调试
            debug_path = Path(self.output_path).with_suffix(".error.txt")
            debug_path.write_text(raw_response, encoding="utf-8")
            print(f"原始错误输出已保存至: {debug_path}")
            raise e
        except Exception as e:
            print(f"Structurer 处理过程中出现错误: {e}")
            raise e


if __name__ == "__main__":
    # 使用示例
    API_KEY = "sk-xxxxxxxx"  # 请替换为实际 Key 或从环境变量获取
    INPUT_FILE = "../data/raw/law_text.txt"
    OUTPUT_FILE = "../deprecated/test_output.json"

    # 确保有测试用的输入文件
    # Path(INPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    # if not Path(INPUT_FILE).exists():
    #     Path(INPUT_FILE).write_text("测试法律文本内容...", encoding="utf-8")

    structurer = Structurer(
        llm_api_key=API_KEY,
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE
    )

    # 执行处理
    structurer.process()