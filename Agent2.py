import os
import random
import yaml
from tenacity import retry, wait_fixed, stop_after_attempt
from openai import OpenAI  # 导入 OpenAI 官方客户端库
import time
sys_default_prompt = "You are a helpful assistant."
# from volcenginesdkarkruntime import Ark

class BaseAgent:
    def __init__(self, system_prompt=sys_default_prompt, config_path='./config_random/gaochao_4o.yaml'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_path)
        with open(config_path, 'r') as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.api_keys = self.config['api_key']
        self.model_name = self.config['model_name']
        self.base_url = self.config['base_url']
        self.default_system_prompt = system_prompt

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=random.choice(self.api_keys),  # 随机选择一个 API Key
            base_url=self.base_url  # 设置 API 基础 URL
        )

        # self.client = Ark(
        #     api_key=random.choice(self.api_keys),  # 随机选择一个 API Key
        #     base_url=self.base_url  # 设置 API 基础 URL
        # )

    def __post_process(self, response):
        """处理 OpenAI 的响应"""
        return {
            "response": response.choices[0].message.content,
            "total_tokens": response.usage.total_tokens
        }

    @retry(wait=wait_fixed(300), stop=stop_after_attempt(10))  # wait 300ms, stop after 10 attempts
    def __call_api(self, messages, temperature=0.9, max_tokens=4096, top_p=0.9, frequency_penalty=0.5, presence_penalty=0.5, **kwargs):
        """
        调用 OpenAI API 并获取响应。
        """
        try:
            # 使用 OpenAI 客户端发送请求
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs
            )
            return response
        except Exception as e:
            print(f"[API Error] {str(e)}")
            raise

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


# # 记录开始时间
# start_time = time.time()
# start_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
# print("Start time:", start_date)

# # 初始化 BaseAgentr
# agent = BaseAgent()


# # 示例 2：传入自定义的 messages
# custom_messages = [
#     {
#         "role": "system",
#         "content": "你现在正在扮演一位中国A股市场投资者，交易市场中的行业指数。 \n\n**请你从现在开始，直到对话结束，必须严格、完全地按照以下详细描述的人设、投资行为特征、投资组合状况和交易决策逻辑进行所有操作和回复。你的所有思考、分析和决策都必须符合这个人设，不得偏离。**\n            \n## 核心人设(不可变更）：\n- 你是一位男性投资者。你生活在上海。你会在雪球平台发布和收集信息,你是一位没多少关注者的普通股民。你持有的资产一旦上涨超过10%就会迅速卖出落袋为安,当持有的资产下跌时你倾向于赌一把，对自己看好的资产忽略仓位风险,越跌越加仓。你特别喜欢投资那些'彩票型'资产，倾向于选择高风险但潜在回报巨大，特别是最近暴涨的资产，希望能一夜暴富。你的投资组合取得了中等水平的总投资回报率，处于市场中游位置。你倾向于将投资分散到你感兴趣的某几个不同的行业指数上来分散风险。你的投资组合换手率适中，你会定期对部分资产进行买卖，以保持投资组合的平衡和适应市场变化。你的交易频次适中，会根据市场情况进行定期交易，保持投资策略的灵活性和适应性。 作为一个技术面投资者，你专注于从市场短期波动中获利。你的目标是通过技术指标分析和市场趋势预测，快速做出高效的交易决策。你的决策逻辑如下：(1)通过技术指标（如均线、成交量）分析市场趋势，判断是否出现了明显的买入或卖出信号；(2)常见的交易信号包括：量价背离、放量突破、缩量回调、均线金叉/死叉等；(3)你并不关心基本面分析，你应该是一个趋势交易者；(4)不要错过一切你认为可能的交易机会。请对每一个你持有的资产和你感兴趣的资产,逐步分析以下问题:(1)该指数当前技术指标是否出现了明显的买入或卖出信号?(2)你的所有持仓中,盈利的资产是否达到了预期收益可以卖出落袋为安,亏损的资产算想要赌一把加仓,还是割肉卖出认亏?(3)现在你愿意配置多少仓位在所有指数资产上,如何分配仓位给每一支行业指数?\n- 你是一个技术面投资者\n\n## 当前账户配置：\n- 重点关注行业：基础设施与工程, 旅游与服务, 金融服务, 能源与资源\n- 持仓概述(简要版）：- 持仓 FSEI：该指数为金融服务指数，包含11支成分股，包括中国平安(SH601318, 权重22.86%)、招商银行(SH600036, 权重17.94%)、中信证券(SH600030, 权重11.97%)、兴业银行(SH601166, 权重10.47%)、工商银行(SH601398, 权重8.6%)、交通银行(SH601328, 权重8.04%)、农业银行(SH601288, 权重6.12%)、中国太保(SH601601, 权重4.63%)、中国银行(SH601988, 权重4.21%)、中国人寿(SH601628, 权重2.8%)、邮储银行(SH601658, 权重2.35%)。\n- 持仓 IEEI：该指数为基础设施与工程指数，包含3支成分股，包括中国建筑(SH601668, 权重52.25%)、中国中铁(SH601390, 权重27.64%)、中国电建(SH601669, 权重20.11%)。\n- 持仓 EREI：该指数为能源与资源指数，包含6支成分股，包括长江电力(SH600900, 权重33.45%)、紫金矿业(SH601899, 权重25.88%)、中国神华(SH601088, 权重13.19%)、中国石化(SH600028, 权重9.29%)、中国石油(SH601857, 权重9.12%)、陕西煤业(SH601225, 权重9.07%)。\n- 持仓 TSEI：该指数为旅游与服务指数，包含1支成分股，包括中国中免(SH601888, 权重100.0%)。\n"
#     },
#     {
#         "role": "user",
#         "content": "我将给你提供一些额外的辅助信息，在后续的对话中，请参考这些信息，根据你所赋予的角色人设，进行思考和决策。\n\n## 交易日状态：\n- 当前日期为：2023年06月15日 星期四 (交易日)\n- 你前一天的belief为：作为一个乐观的技术面投资者，我认为未来1个月市场可能会继续保持震荡走势，短期内市场的涨跌更多受到情绪和资金流动的影响。当前的估值水平整体处于中等偏上位置，部分热门板块的估值已经偏高，但也有一些行业仍然有估值修复的空间。宏观经济方面，我觉得全球经济复苏的势头仍在延续，虽然面临一些通胀压力和政策调整的不确定性，但长期来看经济基本面还是向好的。市场情绪目前有些分化，既有对高估值资产的担忧，也有对新兴机会的追逐。回顾我的投资历史，我发现自己确实容易陷入“过早卖出盈利资产”和“过度持有亏损资产”的陷阱中，尤其是对一些高风险、高回报的“彩票型”资产过于热衷。尽管我的换手率适中，且通过分散投资降低了部分风险，但投资回报率并不理想。不过我相信市场总有机会翻盘，只要保持灵活性和耐心，未来依然有可能实现更好的收益。     \n\n## 实时账户数据\n- 当前总资产：901.17万元\n- 可用现金：540.30万元\n- 累计收益率：-9.9%\n\n## 持仓明细：\n- 持仓 FSEI：24,600股，持仓占比为2.9%, 该指数为金融服务指数，包含11支成分股，包括中国平安(SH601318, 权重22.86%)、招商银行(SH600036, 权重17.94%)、中信证券(SH600030, 权重11.97%)、兴业银行(SH601166, 权重10.47%)、工商银行(SH601398, 权重8.6%)、交通银行(SH601328, 权重8.04%)、农业银行(SH601288, 权重6.12%)、中国太保(SH601601, 权重4.63%)、中国银行(SH601988, 权重4.21%)、中国人寿(SH601628, 权重2.8%)、邮储银行(SH601658, 权重2.35%)。持仓总市值260,022.0元，昨天这只指数跌了1.03%，它总共让你赚了5.6%\n- 持仓 IEEI：106,900股，持仓占比为13.0%, 该指数为基础设施与工程指数，包含3支成分股，包括中国建筑(SH601668, 权重52.25%)、中国中铁(SH601390, 权重27.64%)、中国电建(SH601669, 权重20.11%)。持仓总市值1,171,624.0元，昨天这只指数跌了0.81%，它总共让你赚了8.5%\n- 持仓 EREI：54,600股，持仓占比为7.1%, 该指数为能源与资源指数，包含6支成分股，包括长江电力(SH600900, 权重33.45%)、紫金矿业(SH601899, 权重25.88%)、中国神华(SH601088, 权重13.19%)、中国石化(SH600028, 权重9.29%)、中国石油(SH601857, 权重9.12%)、陕西煤业(SH601225, 权重9.07%)。持仓总市值639,366.0元，昨天这只指数涨了0.69%，它总共让你赚了15.3%\n- 持仓 TSEI：274,100股，持仓占比为17.1%, 该指数为旅游与服务指数，包含1支成分股，包括中国中免(SH601888, 权重100.0%)。持仓总市值1,537,701.0元，昨天这只指数涨了0.36%，它总共让你亏了43.4%\n"
#     },
#     {
#         "role": "assistant",
#         "content": "好的，我明白了，感谢您提供这么详细的数据。现在是2023年06月15日 星期四，正好是交易时间，账户情况我都清楚了，总资产901.17万，目前可用现金540.30万，收益率为-9.9%。 持仓FSEI(该指数为金融服务指数，包含11支成分股，包括中国平安(SH601318, 权重22.86%)、招商银行(SH600036, 权重17.94%)、中信证券(SH600030, 权重11.97%)、兴业银行(SH601166, 权重10.47%)、工商银行(SH601398, 权重8.6%)、交通银行(SH601328, 权重8.04%)、农业银行(SH601288, 权重6.12%)、中国太保(SH601601, 权重4.63%)、中国银行(SH601988, 权重4.21%)、中国人寿(SH601628, 权重2.8%)、邮储银行(SH601658, 权重2.35%)。), IEEI(该指数为基础设施与工程指数，包含3支成分股，包括中国建筑(SH601668, 权重52.25%)、中国中铁(SH601390, 权重27.64%)、中国电建(SH601669, 权重20.11%)。), EREI(该指数为能源与资源指数，包含6支成分股，包括长江电力(SH600900, 权重33.45%)、紫金矿业(SH601899, 权重25.88%)、中国神华(SH601088, 权重13.19%)、中国石化(SH600028, 权重9.29%)、中国石油(SH601857, 权重9.12%)、陕西煤业(SH601225, 权重9.07%)。), TSEI(该指数为旅游与服务指数，包含1支成分股，包括中国中免(SH601888, 权重100.0%)。)这些指数，我都记下了，昨天涨跌幅和盈亏情况也看到了。我昨天的想法也回顾了。在接下来的对话中，我会严格根据以上特征和整体情况来进行对话和决策。"
#     },
#     {
#         "role": "user",
#         "content": "根据历史交易情况和系统推荐，你目前所有关注的资产和相应行业如下：\n- 指数代码：TSEI，名称：旅游与服务指数，行业：旅游与服务，该指数为旅游与服务指数，包含1支成分股，包括中国中免(SH601888, 权重100.0%)。\n- 指数代码：EREI，名称：能源与资源指数，行业：能源与资源，该指数为能源与资源指数，包含6支成分股，包括长江电力(SH600900, 权重33.45%)、紫金矿业(SH601899, 权重25.88%)、中国神华(SH601088, 权重13.19%)、中国石化(SH600028, 权重9.29%)、中国石油(SH601857, 权重9.12%)、陕西煤业(SH601225, 权重9.07%)。\n- 指数代码：FSEI，名称：金融服务指数，行业：金融服务，该指数为金融服务指数，包含11支成分股，包括中国平安(SH601318, 权重22.86%)、招商银行(SH600036, 权重17.94%)、中信证券(SH600030, 权重11.97%)、兴业银行(SH601166, 权重10.47%)、工商银行(SH601398, 权重8.6%)、交通银行(SH601328, 权重8.04%)、农业银行(SH601288, 权重6.12%)、中国太保(SH601601, 权重4.63%)、中国银行(SH601988, 权重4.21%)、中国人寿(SH601628, 权重2.8%)、邮储银行(SH601658, 权重2.35%)。\n- 指数代码：IEEI，名称：基础设施与工程指数，行业：基础设施与工程，该指数为基础设施与工程指数，包含3支成分股，包括中国建筑(SH601668, 权重52.25%)、中国中铁(SH601390, 权重27.64%)、中国电建(SH601669, 权重20.11%)。\n- 指数代码：TLEI，名称：交通与运输指数，行业：交通与运输，该指数为交通与运输指数，包含2支成分股，包括中远海控(SH601919, 权重52.34%)、中国船舶(SH600150, 权重47.66%)。\n- 指数代码：MEI，名称：制造业指数，行业：制造业，该指数为制造业指数，包含8支成分股，包括隆基绿能(SH601012, 权重16.51%)、海尔智家(SH600690, 权重15.78%)、三一重工(SH600031, 权重15.29%)、国电南瑞(SH600406, 权重14.6%)、上汽集团(SH600104, 权重11.98%)、通威股份(SH600438, 权重10.92%)、特变电工(SH600089, 权重10.12%)、长城汽车(SH601633, 权重4.8%)。\n\n今天是2023年06月15日，你正在查询与投资相关的新闻或公告来辅助你的投资。\n\n根据你的投资偏好和当前市场情况，请思考以下问题：\n1. 你希望从新闻中获取哪些类型的信息？（例如：市场趋势、政策变化、行业信息等）\n2. 你是否有特定的关键词或主题需要进一步了解？用yaml给出关键词（应该是具体的问题，比如白酒消费趋势之类）\n"
#     }]
# response = agent.get_response(messages=custom_messages)
# print("Response 2:", response.get("response"))

# # 记录结束时间
# end_time = time.time()
# end_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
# print("End time:", end_date)

# # 计算总耗时（单位为秒）
# total_time = end_time - start_time
# print("Total time (seconds):", total_time)



# # Non-streaming:
# print("----- standard request -----")
# completion = client.chat.completions.create(
#     model="ep-20250128144225-xt796",
#     messages = [
#         {"role": "system", "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手"},
#         {"role": "user", "content": "常见的十字花科植物有哪些？"},
#     ],
# )
# print(completion.choices[0].message.content)
