"""DashScope 流式响应聚合器

专门处理 QVQ 等仅支持流式输出的模型。
QVQ 系列模型特点：
- 仅支持流式输出（incremental_output 强制为 true）
- 输出包含 reasoning_content（思考过程）和 content（最终回答）
- 属于"仅思考模型"，总会在回复前进行思考

参考：vlm.txt 中的 QVQ 使用示例
"""

from __future__ import annotations

import os
from typing import Any

import dashscope


class DashScopeStreamAggregator:
    """DashScope 流式响应聚合器"""

    @staticmethod
    def call_and_aggregate(
        model: str,
        messages: list[dict[str, Any]],
        api_key: str | None = None,
        extract_reasoning: bool = True,  # ✅ 改为默认 True（QVQ 总是思考）
        debug: bool = False,
        **kwargs: Any
    ) -> tuple[str, str]:
        """
        流式调用 DashScope 并聚合响应

        Args:
            model: 模型名称（如 "qvq-max"）
            messages: 消息列表
            api_key: DashScope API Key（默认从环境变量读取）
            extract_reasoning: 是否提取思考过程（默认True，QVQ 总是输出思考）
            debug: 是否输出调试日志
            **kwargs: 额外参数传递给 API

        Returns:
            (reasoning_content, answer_content)
            - reasoning_content: 思考过程（QVQ 总是有内容，约800-2200字符）
            - answer_content: 最终回答（JSON 格式）

        Raises:
            Exception: API 调用失败
        """
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")

        # QVQ 必须使用流式输出
        # 明确设置 incremental_output=True（虽然 QVQ 默认就是 true）
        response = dashscope.MultiModalConversation.call(
            api_key=api_key,
            model=model,
            messages=messages,
            stream=True,  # QVQ 必须开启流式
            incremental_output=True,  # ✅ 明确设置增量输出
            **kwargs
        )

        # 完整思考过程（QVQ 总是输出，约800-2200字符）
        reasoning_content = ""
        # 完整回复（JSON 格式）
        answer_content = ""

        # 统计信息
        chunk_count = 0
        reasoning_chunks = 0
        answer_chunks = 0

        # 遍历流式响应
        for chunk in response:
            chunk_count += 1

            # 检查响应状态
            if chunk.status_code != 200:
                if debug:
                    print(f"[QVQ Debug] Chunk {chunk_count}: status_code={chunk.status_code}")
                continue

            message = chunk.output.choices[0].message

            # 提取思考过程（QVQ 总是有）
            reasoning_chunk = message.get("reasoning_content", None)
            if reasoning_chunk is not None and reasoning_chunk != "":
                reasoning_content += reasoning_chunk
                reasoning_chunks += 1
                if debug:
                    print(f"[QVQ Debug] 思考 chunk {reasoning_chunks}: +{len(reasoning_chunk)} 字符")

            # 提取最终回答（JSON）
            if message.content and len(message.content) > 0:
                for item in message.content:
                    if isinstance(item, dict) and "text" in item:
                        answer_content += item["text"]
                        answer_chunks += 1
                        if debug:
                            print(f"[QVQ Debug] 回答 chunk {answer_chunks}: +{len(item['text'])} 字符")

        # QVQ 总是返回思考过程（即使 extract_reasoning=False）
        if debug or len(reasoning_content) > 0:
            print(f"[QVQ] 思考过程: {len(reasoning_content)} 字符 ({reasoning_chunks} chunks)")
            print(f"[QVQ] 最终回答: {len(answer_content)} 字符 ({answer_chunks} chunks)")

        # 根据参数决定是否返回思考过程
        if extract_reasoning:
            return reasoning_content, answer_content
        else:
            return "", answer_content

    @staticmethod
    def call_and_aggregate_safe(
        model: str,
        messages: list[dict[str, Any]],
        api_key: str | None = None,
        extract_reasoning: bool = True,  # ✅ 改为默认 True
        timeout: int = 120,  # 默认2分钟超时
        **kwargs: Any
    ) -> tuple[str, str, str | None]:
        """
        带异常处理的流式聚合

        Returns:
            (reasoning_content, answer_content, error_msg)
            - error_msg: 如果出错则包含错误信息，否则为 None
        """
        try:
            reasoning, answer = DashScopeStreamAggregator.call_and_aggregate(
                model=model,
                messages=messages,
                api_key=api_key,
                extract_reasoning=extract_reasoning,
                **kwargs
            )
            return reasoning, answer, None
        except Exception as e:
            error_msg = f"流式聚合失败: {str(e)}"
            return "", "", error_msg


# 兼容性别名
StreamAggregator = DashScopeStreamAggregator

