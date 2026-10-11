"""配置加载:读取 .env 中的 DeepSeek 配置,初始化 LLM。"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
#: 当前官方模型名(定价页:https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。
#: 老名字 `deepseek-chat` 已不在官方表里 —— 实测仍可调用、由 V4.1-Flash 提供服务,
#: 但遗留别名哪天消失是不通知的,所以默认值用当前名。
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    """返回配置好的 DeepSeek LLM。未配置 key 时抛出友好错误。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入你的 key。")
    return ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
    )
