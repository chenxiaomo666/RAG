from typing import List

import requests

# DeepSeek 的 OpenAI 兼容接口地址
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"
API_KEY = "*"

SYSTEM_PROMPT = (
    "你是一位只是助手，请根据用户的问题和下列片段生成准确的回答"
)


def chat(user_input: str, chunks: List[str]) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + \
               [{"role": "user", "content": f"用户输入：{user_input}, 相关片段：{chunks}"}]

    """向 DeepSeek 发送请求，返回接口的完整响应 JSON。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
    }
    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
    return resp.json()

