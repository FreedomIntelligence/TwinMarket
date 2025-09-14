"""
AI代理基础类模块

该模块定义了一个通用的AI代理基础类，用于与OpenAI API进行交互。
支持多种AI模型、重试机制、并发API调用等功能。
"""

# 标准库导入
import os
import random
import time

# 第三方库导入
import yaml
from openai import OpenAI  # OpenAI官方客户端库
from tenacity import retry, wait_fixed, stop_after_attempt  # 重试机制库

# ============================ 全局配置 ============================
# 默认系统提示词
sys_default_prompt = "You are a helpful assistant."



class BaseAgent:
    """
    AI代理基础类
    
    该类封装了与OpenAI API的交互逻辑，提供了统一的接口来调用各种语言模型。
    支持多个API密钥的随机选择、重试机制、参数可配置等特性。
    
    Attributes:
        config: 从配置文件加载的配置信息
        api_keys: API密钥列表
        model_name: 使用的模型名称
        base_url: API基础URL
        default_system_prompt: 默认系统提示词
        client: OpenAI客户端实例
    """
    
    def __init__(self, system_prompt=sys_default_prompt, config_path='./config_random/zyf.yaml'):
        """
        初始化AI代理
        
        Args:
            system_prompt (str): 系统提示词，定义AI的角色和行为模式
            config_path (str): 配置文件路径，包含API密钥、模型名称等信息
        """
        # 获取配置文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_path)
        
        # 加载配置文件
        with open(config_path, 'r') as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)
            
        # 提取配置信息
        self.api_keys = self.config['api_key']        # API密钥列表
        self.model_name = self.config['model_name']   # 模型名称
        self.base_url = self.config['base_url']       # API基础URL
        self.default_system_prompt = system_prompt    # 默认系统提示词

        # 初始化OpenAI客户端，随机选择一个API密钥
        self.client = OpenAI(
            api_key=random.choice(self.api_keys),  # 从多个密钥中随机选择
            base_url=self.base_url                 # 设置API基础URL
        )

    def __post_process(self, response):
        """
        处理OpenAI API的响应数据
        
        从原始API响应中提取有用信息，包括生成的文本内容和使用的token数量。
        
        Args:
            response: OpenAI API返回的原始响应对象
            
        Returns:
            dict: 包含响应内容和token使用情况的字典
        """
        return {
            "response": response.choices[0].message.content,  # 提取AI生成的文本内容
            "total_tokens": response.usage.total_tokens      # 提取总的token使用数量
        }

    @retry(wait=wait_fixed(1000), stop=stop_after_attempt(10))  
    def __call_api(self, messages, temperature=0.9, max_tokens=8192, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5, **kwargs):
        """
        调用OpenAI API并获取响应
        
        使用tenacity库实现重试机制，在API调用失败时会自动重试。
        重试间隔为1秒，最多重试10次。
        
        Args:
            messages (list): 对话消息列表
            temperature (float): 温度参数，控制输出的随机性
            max_tokens (int): 最大生成token数量
            top_p (float): 核采样参数
            frequency_penalty (float): 频率惩罚参数
            presence_penalty (float): 存在惩罚参数
            **kwargs: 其他可选参数
            
        Returns:
            response: OpenAI API的原始响应对象
            
        Raises:
            Exception: 当API调用失败时抛出异常
        """
        try:
            # 使用OpenAI客户端发送聊天完成请求
            response = self.client.chat.completions.create(
                model=self.model_name,              # 模型名称
                messages=messages,                  # 对话消息
                temperature=temperature,            # 控制输出的创意性
                max_tokens=max_tokens,              # 最大生成长度
                top_p=top_p,                        # 核采样参数
                frequency_penalty=frequency_penalty, # 频率惩罚
                presence_penalty=presence_penalty,   # 存在惩罚
                **kwargs                            # 其他参数
            )
            return response
        except Exception as e:
            # 记录API错误信息
            print(f"[API错误] {str(e)}")
            raise  # 重新抛出异常，触发重试机制

    def get_response(
        self,
        user_input=None,
        system_prompt=None,
        temperature=0.9,
        max_tokens=4096,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        debug=False,
        messages=None,  # 新增 messages 参数
        **kwargs
    ):
        """
        获取AI的响应，支持多种输入模式
        
        该方法是主要的对外接口，支持传入单个用户输入或完整的对话消息列表。
        具有灵活的参数配置和错误处理机制。
        
        Args:
            user_input (str, optional): 用户输入的文本内容
            system_prompt (str, optional): 系统提示词，默认使用初始化时的提示词
            temperature (float): 温度参数，控制输出的随机性 (0-1)
            max_tokens (int): 最大生成token数量
            top_p (float): 核采样参数 (0-1)
            frequency_penalty (float): 频率惩罚参数 (-2.0-2.0)
            presence_penalty (float): 存在惩罚参数 (-2.0-2.0)
            debug (bool): 是否开启调试模式，会打印响应内容
            messages (list, optional): 完整的对话消息列表
            **kwargs: 其他传递给API的参数
            
        Returns:
            dict: 包含响应内容和token使用情况的字典，
                  或在出错时返回包含error字段的字典
        """
        try:
            # 使用默认系统提示词（如果未提供）
            if system_prompt is None:
                system_prompt = self.default_system_prompt

            # 初始化消息列表（如果未提供）
            if messages is None:
                messages = []

            # 检查并添加系统提示词（如果不存在）
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

            # 添加用户输入到消息列表（如果提供了user_input）
            if user_input is not None:
                messages.append({"role": "user", "content": user_input})

            # 清理kwargs中的messages参数，避免参数冲突
            if "messages" in kwargs:
                kwargs.pop("messages")

            # 调用底层API接口
            response = self.__call_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs
            )

            # 处理API响应，提取有用信息
            result = self.__post_process(response)

            # 调试模式：以绿色文字打印响应内容
            if debug:
                print("\033[92m" + f"[响应] {result['response']}" + "\033[0m")

            # 返回处理后的结果
            return result

        except Exception as e:
            # 错误处理：以红色文字打印错误信息
            print("\033[91m" + f"[错误] {str(e)}" + "\033[0m")
            return {"error": f"Error: {str(e)}"}

# ============================ 模块初始化和测试 ============================
# 记录模块加载的开始时间
start_time = time.time()
start_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print("模块加载开始时间:", start_date)