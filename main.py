import json
import config
from utils.structuring import StructuringLaw


if __name__ == "__main__":
    structurer = StructuringLaw(
        api_key=config.api_key,
        base_url=config.base_url,
        model_name=config.model_name,
        prompt_path="prompts/prompt_chunk.txt",
        output_path="data/json/test_output.json"
    )

    with open("data/law/中华人民共和国劳动法.txt","r",encoding = 'utf-8') as f:
        law_text = f.read()

    result = structurer.process(law_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))