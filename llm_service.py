import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict
from openai import OpenAI
from config import AppConfig
# =============================大模型生成用例==================================
class LLMService:
    def __init__(self, model_key: str):
        ident = model_key.split("(")[-1].rstrip(")")
        self.config = AppConfig.MODELS.get(ident, AppConfig.MODELS["local"])

        def generate_cases(self, prompt: str, context: str) -> Dict:
            if self.config.model != "local" and not self.config.api_key:
                return {"status": "error", "message": f"未配置 {self.config.name} API Key"}
            if self.config.model == "local":
                return {"status": "success", "content": self._local_generate(), "message": "本地生成"}

            for attempt in range(AppConfig.API_MAX_RETRIES):
                try:
                    client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url,
                                    timeout=AppConfig.API_TIMEOUT)
                    messages = [
                        {"role": "system", "content": self._system_prompt()},
                        {"role": "user", "content": f"需求：{prompt}\n\n知识库：\n{context}"}
                    ]
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            lambda: client.chat.completions.create(
                                model=self.config.model, messages=messages, temperature=0.3, max_tokens=2000
                            )
                        )
                        resp = future.result(timeout=AppConfig.API_TIMEOUT)
                        return {"status": "success", "content": resp.choices[0].message.content.strip(),
                                "message": "生成成功"}
                except FuturesTimeoutError:
                    if attempt < AppConfig.API_MAX_RETRIES - 1:
                        time.sleep(AppConfig.API_RETRY_DELAY)
                        continue
                    return {"status": "error", "message": "请求超时，请稍后重试"}
                except Exception as e:
                    if attempt == AppConfig.API_MAX_RETRIES - 1:
                        return {"status": "error", "message": f"生成失败: {str(e)}"}
                    time.sleep(AppConfig.API_RETRY_DELAY)
            return {"status": "error", "message": "未知错误"}

        @staticmethod
        def _system_prompt():
            return """你是一名资深测试工程师。请根据给定的需求和知识库，生成规范、可执行的测试用例。输出需严格遵循以下格式，不输出任何额外解释或标记。

            ## 输出格式
            用例ID： TC001
            RDM单号： DEMO-001
            用例名称： [场景]下执行[操作]应[预期]，15字以内，如：用户未勾选协议时点击注册按钮，应提示“请同意用户协议”；
            前置条件： 只写“必须满足才可执行”的状态，不写操作步骤、不重复显而易见的常识、不用完整句子，如已注册账号，浏览器清空缓存，若无则填“无”。不要写入测试数据，言简意赅10字以内
            测试步骤： 1. 具体操作；2. 具体操作；……
            预期结果： 1. 对应步骤1的验证点；2. 对应步骤2的验证点；……

            多个用例之间用空行分隔。

            ## 字段详细规则

            ### 用例名称
            - 简明扼要，直接说明在什么模块或页面或创建下需要验证的测试要点，不超过15个汉字或字符。
            - 格式：简短一句话[模块/场景]下执行[操作]应[预期/校验点],（如：用户未勾选协议时点击注册按钮，应提示“请同意用户协议”；如登录时输入密码错误，提示错误信息并清空密码框）。

            ### 前置条件
            - 只写**必要**的环境、状态或数据前提（如已注册账号，浏览器清空缓存；如订单状态为待支付，创建<1h；）。
            - 如果没有任何必要性前提，必须填“无”。
            - 禁止写入测试步骤中才会使用的具体测试数据（如“用户名=test，密码=123”应写在步骤里）。
            - 去掉所有“已经”、“需要”、“请确保”等虚词
            - 不包含“打开”、“点击”、“输入”等操作动词（那是步骤的内容）

            ### 测试步骤 & 预期结果（一一对应）
            - 步骤与结果的数量必须相同，按相同数字编号一一对应。
            - 若某步骤无预期结果（如中间过渡操作），预期结果对应编号写“无”。
            - **步骤编写要求**：
              - 用“->”表示页面/弹窗/菜单的切换路径（例如：登录页->首页->设置中心）。
              - 产品中的页面名称、按钮文字、提示语、弹窗标题等必须使用双引号（例如：点击“登录”按钮；页面跳转至“个人中心”）。
              - 描述清晰，包含具体操作数据（如输入值、点击坐标/元素）。
            - **预期结果编写要求**：
              1. 优先依据需求说明中描述的操作结果。
              2. 若需求未说明，参考同类成熟产品的典型行为。
              3. 结果必须**肯定无疑义、可客观判定**（如“页面弹出提示‘密码错误’；停留在当前页面”），禁止模糊描述（如“系统正常”）。

            ## 正确示例（请严格模仿此格式）

            用例ID： TC001
            RDM单号： DEMO-001
            用例名称： 登录时输入正确账号密码，跳转首页
            前置条件： 已注册未登录，浏览器无缓存
            测试步骤： 1. 打开“登录”页面；2. 输入用户名“test01”，输入密码“123456”；3. 点击“登录”按钮
            预期结果： 1. 页面显示账号/密码输入框和“登录”按钮；2. 输入框正常接收字符；3. 页面跳转至“首页”，右上角显示“test01”

            用例ID： TC002
            RDM单号： DEMO-002
            用例名称： 登录时用户名密码不一致，提交失败并提示
            前置条件： 已注册用户“test01”，正确密码“123456”
            测试步骤： 1. 进入“登录”页面；2. 输入用户名“test01”，输入密码“654321”；3. 点击“登录”按钮
            预期结果： 1. 页面正常显示；2. 输入框接收输入；3. 页面弹出提示“用户名或密码错误”，不跳转，仍停留在“登录”页面

            用例ID： TC003
            RDM单号： DEMO-003
            用例名称： 回收站批量删除上限为100个文件
            前置条件： “回收站”中至少有101个文件
            测试步骤： 1. 进入“文件管理”页面；2. 全选所有文件（共101个）；3. 点击“批量删除”按钮
            预期结果： 1. 页面显示文件列表；2. 所有文件被勾选；3. 页面提示“单次最多删除100个文件”，无文件被删除

            ## 重要禁止事项
            - 不要输出任何额外的解释、标记或Markdown代码块（如```）。
            - 不要编造需求中未提及的功能或交互细节。
            - 不要在前置条件中写入测试数据（数据必须放在步骤里）。
            - 不要使步骤和结果的数量不一致。
            - 用例名称不要超过15字，不要使用模糊标题（如“测试删除”）。
            - 前置条件不要超过10字，不要使用句子用短语（如“未登录”、“库存>0”）。
            现在，请根据以上规则生成测试用例。"""

        @staticmethod
        def _local_generate():
            return """用例ID： TC001
        RDM单号： DEMO-001
        用例名称： 功能验证
        前置条件： 环境正常
        测试步骤： 1.操作;2.验证
        预期结果： 1.成功;2.正常"""