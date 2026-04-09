from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    api_key: str
    base_url: str
    model: str

class AppConfig:
    PAGE_TITLE = "测试用例智构系统"
    LAYOUT = "wide"
    BASE_ROOT = "knowledge_projects"        # OSS 中的根目录前缀
    ALLOWED_FILE_TYPES = {
        "txt", "csv", "docx", "doc", "pdf", "xlsx", "xls", "pptx", "xmind", "md"
    }
    RAG_TOP_K = 5  #检索返回最大片段数量
    RAG_SIMILARITY_THRESHOLD = 0.2  # 检索返回最低相似度
    API_TIMEOUT = 60
    API_MAX_RETRIES = 2
    API_RETRY_DELAY = 2

    MODELS = {
        "qwen": ModelConfig(
            name="通义千问",
            api_key="sk-66b40867237b4e589c81c0255ff94b36",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus"
        ),
        "local": ModelConfig(
            name="本地离线",
            api_key="none",
            base_url="none",
            model="local"
        )
    }

    @classmethod
    def get_model_list(cls):
        return [f"{m.name}({k})" for k, m in cls.MODELS.items() if m.api_key or k == "local"]