"""Persian analysis prompt templates for MCP clients."""

from __future__ import annotations


def persian_text_analysis_prompt(text: str) -> str:
    """Guide an agent to analyze Persian text through Rakhshai graph tools."""

    return (
        "این متن فارسی را "
        "با روش گراف‌محور تحلیل کن. "
        "ابتدا `rakhshai_analyze_persian_text` را اجرا کن، "
        "سپس کلمات کلیدی، "
        "موجودیت‌های احتمالی، "
        "گره‌های مهم "
        "و رابطه‌های اصلی را به زبان فارسی "
        f"توضیح بده.\n\nمتن:\n{text}"
    )


def graph_memory_generation_prompt(prompt: str) -> str:
    """Guide graph-memory generation with explicit evidence."""

    return (
        "با استفاده از `rakhshai_graph_memory_generate` پاسخ بساز. "
        "خروجی باید مفاهیم مرکزی، "
        "شواهد بازیابی‌شده از حافظه "
        "و مسیر استدلال گرافی را "
        f"نشان دهد.\n\nدرخواست:\n{prompt}"
    )


def research_report_prompt(topic: str) -> str:
    """Guide a compact research-style graph-NLP report."""

    return (
        "برای موضوع زیر یک گزارش پژوهشی کوتاه بساز: "
        "تحلیل متنی، گراف مفهومی، "
        "خلاصه، گره‌های کلیدی، "
        "محدودیت‌ها و پیشنهاد آزمایش بعدی.\n\n"
        f"موضوع:\n{topic}"
    )


def model_comparison_prompt(model_a: str, model_b: str) -> str:
    """Guide an MCP client to compare two Rakhshai model runs."""

    return (
        "با resourceهای `rakhshai://models` و `rakhshai://runs` "
        "دو مدل را مقایسه کن "
        "و فقط بر اساس metadata و metrics "
        "قابل خواندن نتیجه بگیر.\n\n"
        f"مدل اول: {model_a}\nمدل دوم: {model_b}"
    )


def explainable_nlp_prompt(text: str) -> str:
    """Guide explainable graph-NLP output."""

    return (
        "`rakhshai_explain_result` را برای متن زیر اجرا کن "
        "و توضیح بده خروجی "
        "بر اساس کدام گره‌ها، رابطه‌ها "
        "و مسیرهای گرافی قابل دفاع است.\n\n"
        f"متن:\n{text}"
    )
