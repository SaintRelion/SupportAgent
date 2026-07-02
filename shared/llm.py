from langchain_openai import ChatOpenAI
from shared.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, MODEL


def get_llm(
    temperature: float = 0.3,
    model_override: str = None,
    reasoning_effort: str = None,
) -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI client pointed at OpenRouter.

    model_override: force a specific model regardless of ENV setting.
                    Used for cheap evaluation calls (deepseek) even in prod.
    reasoning_effort: "low" | "medium" | "high" | "max" | "x-high" — passed
                    through to models that support adaptive reasoning (e.g.
                    Sonnet 5). Lower effort reduces internal reasoning tokens
                    and therefore cost. Ignored by models that don't support it.
    """
    extra_body = {}
    if reasoning_effort:
        extra_body["reasoning"] = {"effort": reasoning_effort}

    return ChatOpenAI(
        model=model_override or MODEL,
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "https://your-company.com",
            "X-Title": "Discord Automation",
        },
        extra_body=extra_body or None,
    )