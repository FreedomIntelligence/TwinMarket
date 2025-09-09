import time
import json
import random
import os
import yaml
import requests
from tenacity import retry, wait_fixed, stop_after_attempt

sys_default_prompt = "You are a helpful assistant."


class BaseAgent:
    def __init__(self, system_prompt=sys_default_prompt, config_path='./config_random/gaochao_4o_mini.yaml'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_path)
        with open(config_path, 'r') as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.api_keys = self.config['api_key']
        self.model_name = self.config['model_name']
        self.base_url = self.config['base_url']
        self.default_system_prompt = system_prompt

    def __post_process(self, response):
        """处理 OpenAI 的响应"""
        return {
            "response": response["choices"][0]["message"]["content"],
            "total_tokens": response["usage"]["total_tokens"]
        }

    @retry(wait=wait_fixed(300), stop=stop_after_attempt(10))  # wait 300ms, stop after 50 attempts
    def __call_api(self, messages, temperature=0.9, max_tokens=8192, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5, **kwargs):
        """
        调用 OpenAI API 并获取响应。
        """
        # 随机选择一个 API Key
        api_key = random.choice(self.api_keys)

        # 设置请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 设置请求体
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        payload.update(kwargs)  # 合并其他可选参数

        # 发送请求并获取响应
        response =  requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=10  # 设置超时时间为 300 秒
        )
        response.raise_for_status()  # 如果请求失败，抛出异常
        return response.json()

    def get_response(
        self,
        user_input=None,
        system_prompt=None,
        temperature=0.9,
        max_tokens=8192,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        debug=False,
        messages=None,  # 新增 messages 参数
        **kwargs
    ):
        """
        获取 OpenAI 的响应，支持传入 messages 参数。
        """
        try:
            if system_prompt is None:
                system_prompt = self.default_system_prompt

            # 构建消息列表
            if messages is None:
                messages = []

            # 添加系统提示（如果不存在）
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

            # 添加用户输入（如果提供了 user_input）
            if user_input is not None:
                messages.append({"role": "user", "content": user_input})

            # 确保 kwargs 中不包含 messages
            if "messages" in kwargs:
                kwargs.pop("messages")

            # 调用 API
            response = self.__call_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs
            )

            # 处理响应
            result = self.__post_process(response)

            # 打印调试信息
            if debug:
                print("\033[92m" + f"[Response] {result['response']}" + "\033[0m")

            # 返回响应内容
            return result

        except Exception as e:
            # 打印错误信息
            print("\033[91m" + f"[Error] {str(e)}" + "\033[0m")
            return {"error": f"Error: {str(e)}"}



