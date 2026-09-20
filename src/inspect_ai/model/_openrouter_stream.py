"""OpenRouter's reasoning_details stream, including server-executed web tools."""

import json
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionChunk

from inspect_ai._util.content import Content, ContentToolUse

from ._openai import openai_chat_completion_stream_final
from ._openrouter_reasoning import openrouter_reasoning_details_to_reasoning
from ._stream import (
    StreamReasoningEvent,
    StreamTextEvent,
    model_stream_partial_requested,
    report_model_stream_content,
)

SERVER_TOOL_DETAIL = "openrouter_server_tool_detail"
WEB_TOOLS = {"openrouter:web_search", "openrouter:web_fetch", "web_search", "web_fetch"}


def server_tool_content(
    detail: dict[str, Any], fallback_id: str
) -> ContentToolUse | None:
    """Project a provider-executed web call without turning it into a client tool."""
    if (
        detail.get("type") != "reasoning.server_tool_call"
        or detail.get("tool_name") not in WEB_TOOLS
    ):
        return None
    return ContentToolUse(
        id=detail.get("tool_call_id") or detail.get("id") or fallback_id,
        tool_type="web_search",
        name=detail["tool_name"].removeprefix("openrouter:"),
        arguments=detail.get("arguments", ""),
        result=detail.get("result", ""),
        internal={SERVER_TOOL_DETAIL: detail},
    )


def reasoning_details_content(details: list[dict[str, Any]]) -> list[Content]:
    """Keep reasoning and web calls ordered, with their original replay records."""
    content: list[Content] = []
    reasoning: list[dict[str, Any]] = []
    for index, detail in enumerate(details):
        tool = server_tool_content(detail, f"openrouter-web-{index}")
        if tool is None:
            reasoning.append(detail)
        else:
            if reasoning:
                content.append(openrouter_reasoning_details_to_reasoning(reasoning))
                reasoning = []
            content.append(tool)
    if reasoning:
        content.append(openrouter_reasoning_details_to_reasoning(reasoning))
    return content


def search_sources_content(annotations: Any, id: str) -> ContentToolUse | None:
    """Expose citation sources when the router does not expose individual calls."""
    if not isinstance(annotations, list):
        return None
    sources = []
    for annotation in annotations:
        raw = (
            annotation.model_dump() if hasattr(annotation, "model_dump") else annotation
        )
        if isinstance(raw, dict) and raw.get("type") == "url_citation":
            source = raw.get("url_citation")
            if isinstance(source, dict) and isinstance(source.get("url"), str):
                sources.append(source)
    return (
        ContentToolUse(
            id=id,
            tool_type="web_search",
            name="search_sources",
            arguments="",
            result=json.dumps(sources),
        )
        if sources
        else None
    )


async def openrouter_stream_final(
    stream: AsyncIterable[ChatCompletionChunk],
) -> ChatCompletion:
    """Accumulate indexed reasoning fragments without concatenating their metadata.

    The OpenAI accumulator does not know OpenRouter's reasoning_details schema.
    Keep those records separately so ids, types and formats survive replay while
    text, signatures, encrypted data and tool argument/result fragments append.
    """
    details: dict[int, dict[str, dict[str, Any]]] = {}

    async def chunks() -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in stream:
            chunk = chunk.model_copy(deep=True)
            for choice in chunk.choices:
                delta = choice.delta
                raw_details = (delta.model_extra or {}).pop("reasoning_details", None)
                display = choice.index == 0 and model_stream_partial_requested()
                readable = False
                if raw_details is not None and not isinstance(raw_details, list):
                    raise ValueError("OpenRouter reasoning_details must be a list")
                if isinstance(raw_details, list):
                    records = details.setdefault(choice.index, {})
                    for raw in raw_details:
                        if not isinstance(raw, dict):
                            raise ValueError(
                                "OpenRouter reasoning_details entries must be objects"
                            )
                        key = (
                            f"index:{raw['index']}"
                            if raw.get("index") is not None
                            else f"id:{raw['id']}"
                            if raw.get("id") is not None
                            else f"item:{len(records)}"
                        )
                        new_record = key not in records
                        record = records.setdefault(key, {})
                        had_result = bool(record.get("result"))
                        for field, value in raw.items():
                            if field in {
                                "text",
                                "summary",
                                "data",
                                "signature",
                                "arguments",
                                "result",
                            } and isinstance(value, str):
                                record[field] = record.get(field, "") + value
                            elif value is not None:
                                if (
                                    field
                                    in {
                                        "id",
                                        "type",
                                        "format",
                                        "tool_call_id",
                                        "tool_name",
                                    }
                                    and field in record
                                    and record[field] != value
                                ):
                                    raise ValueError(
                                        f"OpenRouter changed {field} for reasoning record {key}"
                                    )
                                record[field] = value
                        if display:
                            tool = server_tool_content(record, f"openrouter-web-{key}")
                            if tool is not None:
                                report_model_stream_content(
                                    tool,
                                    force=new_record
                                    or (not had_result and bool(record.get("result"))),
                                )
                            else:
                                text = raw.get("text") or raw.get("summary")
                                if isinstance(text, str):
                                    report_model_stream_content(
                                        StreamReasoningEvent(reasoning=text)
                                    )
                                    readable = True
                if display:
                    sources = search_sources_content(
                        getattr(delta, "annotations", None),
                        f"{chunk.id}-sources-{choice.index}",
                    )
                    if sources is not None:
                        report_model_stream_content(sources)
                    reasoning = getattr(delta, "reasoning", None) or getattr(
                        delta, "reasoning_content", None
                    )
                    if not readable and isinstance(reasoning, str):
                        report_model_stream_content(
                            StreamReasoningEvent(reasoning=reasoning)
                        )
                    if delta.content:
                        report_model_stream_content(StreamTextEvent(text=delta.content))
            yield chunk

    completion = await openai_chat_completion_stream_final(
        chunks(), publish_partial=False
    )
    for choice in completion.choices:
        records = details.get(choice.index)
        if records:
            extra = choice.message.model_extra
            assert extra is not None
            extra["reasoning_details"] = list(records.values())
    return completion
