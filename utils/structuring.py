import json
from pathlib import Path

import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class StructuringLaw:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model_name: str = "Pro/moonshotai/Kimi-K2.5",
        prompt_path: str = "prompts/prompt_chunk.txt",
        output_path: str = "data/structured_output.json"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.prompt_path = prompt_path
        self.output_path = output_path

        self.model = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model_name
        )

        prompt_text = Path(self.prompt_path).read_text(encoding="utf-8")
        self.prompt = ChatPromptTemplate.from_template(prompt_text)

    def get_json(self, law_context: str):
        """
        Convert legal documents into structured JSON
        :param law_context: Legal document
        :return: json,list[dict]
        """
        messages = self.prompt.format_messages(context=law_context)

        full_content = ""
        print("model >> ：", flush=True)

        for chunk in self.model.stream(messages):
            chunk_text = chunk.content if chunk.content else ""
            print(chunk_text, end="", flush=True)
            full_content += chunk_text

        print("\n << model", flush=True)

        content = full_content.strip()

        if content.startswith("```json"):
            content = content[7:].strip()
        if content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        return json.loads(content)

    def save_json(self, data):
        """
        save json to output_path
        """
        output_file = Path(self.output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def process(self, law_context: str):
        """
        structuring and saving
        """
        data = self.get_json(law_context)
        self.save_json(data)
        return data