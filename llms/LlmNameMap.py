from llms.zhipu.ZhipuModel import ZhipuModel
from llms.ollama.ollamaModel import OllamaModel

llm_map_name_to_model = {
    "zhipu": ZhipuModel,
    "ollama": OllamaModel,
}
# print(llm_map_name_to_model["qianfan"])
