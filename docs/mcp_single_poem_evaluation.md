# Single Poem Rakhshai MCP Evaluation

## Goal

بررسی اینکه آیا اتصال Rakhshai MCP به `gpt-5.4` باعث پاسخ بهتر در تحلیل یک شعر فارسی می‌شود یا نه، و اگر بهتر شد چند برابر/چند درصد بهتر شده است.

## Claim Under Test

در این تست تک‌نمونه‌ای، Rakhshai MCP با دادن evidence گرافی واقعی به مدل، کیفیت فهم/تحلیل شعر فارسی را بهتر می‌کند.

## Model And Settings

- Model: `gpt-5.4`
- temperature: `0.0`
- top_p: `1.0`
- max_output_tokens: `3000`
- seed: `42` if available
- Source JSON: `runs/openai_mcp_eval/single_poem_mcp_eval_gpt-5.4_1782989988.json`

### Applied Request Parameters

```json
{
  "direct": {
    "requested": {
      "temperature": 0.0,
      "top_p": 1.0,
      "max_output_tokens": 3000,
      "seed": 42
    },
    "applied": {
      "temperature": 0.0,
      "top_p": 1.0,
      "max_output_tokens": 3000,
      "seed": null
    },
    "fallback_notes": [
      "The OpenAI Responses API rejected `seed` for this model/client, so the request was retried without seed while keeping both conditions identical."
    ]
  },
  "rakhshai_mcp": {
    "requested": {
      "temperature": 0.0,
      "top_p": 1.0,
      "max_output_tokens": 3000,
      "seed": 42
    },
    "applied": {
      "temperature": 0.0,
      "top_p": 1.0,
      "max_output_tokens": 3000,
      "seed": null
    },
    "fallback_notes": [
      "The OpenAI Responses API rejected `seed` for this model/client, so the request was retried without seed while keeping both conditions identical."
    ]
  }
}
```

## Input

### Question

```text
این شعر از دید تو چه معنی می‌دهد؟
معنی اصلی شعر را ساده توضیح بده و بگو رابطه میان آینه، سایه، نام، چراغ، رود، تشنگی، خانه و کلید چیست؟
```

### Poem

```text
در آینه، سایه‌ام از من قدیمی‌تر بود
و نامم از دهانِ پنجره به باران می‌ریخت

چراغی که خوابِ خاکستر می‌دید
راه را به پای گم‌شده‌ام نشان نمی‌داد

رود از کنار من گذشت
اما تشنگی در مشت‌هایم لانه کرده بود

گفتم: کدام سو خانه است؟
باد، کلیدی زنگ‌زده را در سکوت چرخاند
```

## Conditions

### Condition A: direct

```text
به این سوال درباره شعر پاسخ بده.

سوال:
این شعر از دید تو چه معنی می‌دهد؟
معنی اصلی شعر را ساده توضیح بده و بگو رابطه میان آینه، سایه، نام، چراغ، رود، تشنگی، خانه و کلید چیست؟

شعر:
در آینه، سایه‌ام از من قدیمی‌تر بود
و نامم از دهانِ پنجره به باران می‌ریخت

چراغی که خوابِ خاکستر می‌دید
راه را به پای گم‌شده‌ام نشان نمی‌داد

رود از کنار من گذشت
اما تشنگی در مشت‌هایم لانه کرده بود

گفتم: کدام سو خانه است؟
باد، کلیدی زنگ‌زده را در سکوت چرخاند
```

### Condition B: rakhshai_mcp

```text
به این سوال درباره شعر پاسخ بده.
از شواهد گرافی Rakhshai MCP فقط وقتی استفاده کن که مرتبط است.
در پاسخ توضیح بده کدام گره‌ها یا رابطه‌ها کمک کردند.

سوال:
این شعر از دید تو چه معنی می‌دهد؟
معنی اصلی شعر را ساده توضیح بده و بگو رابطه میان آینه، سایه، نام، چراغ، رود، تشنگی، خانه و کلید چیست؟

شعر:
در آینه، سایه‌ام از من قدیمی‌تر بود
و نامم از دهانِ پنجره به باران می‌ریخت

چراغی که خوابِ خاکستر می‌دید
راه را به پای گم‌شده‌ام نشان نمی‌داد

رود از کنار من گذشت
اما تشنگی در مشت‌هایم لانه کرده بود

گفتم: کدام سو خانه است؟
باد، کلیدی زنگ‌زده را در سکوت چرخاند

Rakhshai MCP graph evidence:
{
  "analysis": {
    "status": "success",
    "task": "persian_text_analysis",
    "summary": "Persian text analyzed with Rakhshai graph signals.",
    "keywords": [
      {
        "text": "آینه",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "سایه",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "قدیمی",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "نامم",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "دهان",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "پنجره",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "باران",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "ریخت",
        "score": 0.030303,
        "count": 1
      }
    ],
    "top_nodes": [
      {
        "id": 23,
        "label": "گم‌شده‌ام",
        "type": "token",
        "score": 1.0,
        "weighted_degree": 33.391252
      },
      {
        "id": 18,
        "label": "خواب",
        "type": "token",
        "score": 0.981621,
        "weighted_degree": 32.777547
      },
      {
        "id": 37,
        "label": "سو",
        "type": "token",
        "score": 0.981621,
        "weighted_degree": 32.777547
      },
      {
        "id": 41,
        "label": "کلیدی",
        "type": "token",
        "score": 0.958483,
        "weighted_degree": 32.004958
      },
      {
        "id": 12,
        "label": "دهان",
        "type": "token",
        "score": 0.934198,
        "weighted_degree": 31.194028
      },
      {
        "id": 27,
        "label": "کنار",
        "type": "token",
        "score": 0.934198,
        "weighted_degree": 31.194028
      },
      {
        "id": 38,
        "label": "خانه",
        "type": "token",
        "score": 0.926679,
        "weighted_degree": 30.942974
      },
      {
        "id": 22,
        "label": "پای",
        "type": "token",
        "score": 0.898587,
        "weighted_degree": 30.004958
      }
    ],
    "important_relations": [
      {
        "source": 12,
        "target": 13,
        "source_label": "دهان",
        "target_label": "پنجره",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 13,
        "target": 12,
        "source_label": "پنجره",
        "target_label": "دهان",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 14,
        "target": 15,
        "source_label": "باران",
        "target_label": "می‌ریخت",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 15,
        "target": 14,
        "source_label": "می‌ریخت",
        "target_label": "باران",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 18,
        "target": 19,
        "source_label": "خواب",
        "target_label": "خاکستر",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 19,
        "target": 18,
        "source_label": "خاکستر",
        "target_label": "خواب",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 19,
        "target": 20,
        "source_label": "خاکستر",
        "target_label": "می‌دید",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 20,
        "target": 19,
        "source_label": "می‌دید",
        "target_label": "خاکستر",
        "weight": 3.110434,
        "relation": "pmi"
      }
    ],
    "reasoning_path": [
      "دهان -[pmi]-> پنجره",
      "پنجره -[pmi]-> دهان",
      "باران -[pmi]-> می‌ریخت",
      "می‌ریخت -[pmi]-> باران",
      "خواب -[pmi]-> خاکستر"
    ],
    "graph_memory_report": {},
    "focus_terms": [],
    "focus_nodes": [],
    "focus_relations": [],
    "graph_metrics": {
      "node_count": 52,
      "edge_count": 532,
      "directed": false,
      "relation_edge_counts": {
        "cooccurrence": 166,
        "pmi": 166,
        "dependency": 86,
        "stem": 0,
        "subword": 0,
        "word_document": 90,
        "topic_document": 24
      },
      "node_type_counts": {
        "token": 45,
        "document": 1,
        "topic": 6
      },
      "enabled_relations": [
        "cooccurrence",
        "pmi",
        "dependency",
        "stem",
        "subword",
        "word_document",
        "topic_document"
      ]
    }
  },
  "graph_summary": {
    "status": "success",
    "task": "graph_summarization",
    "summary": "در آینه، سایه‌ام از من قدیمی‌تر بود\nو نامم از دهانِ پنجره به باران می‌ریخت\n\nچراغی که خوابِ خاکستر می‌دید\nراه را به پای گم‌شده‌ام نشان نمی‌داد\n\nرود از کنار من گذشت\nاما تشنگی در مشت‌هایم لانه کرده بود\n\nگفتم: کدام سو خانه است \nباد، کلیدی زنگ‌زده را در سکوت چرخاند",
    "keywords": [
      {
        "text": "آینه",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "سایه",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "قدیمی",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "نامم",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "دهان",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "پنجره",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "باران",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "ریخت",
        "score": 0.030303,
        "count": 1
      }
    ],
    "top_nodes": [
      {
        "id": 23,
        "label": "گم‌شده‌ام",
        "type": "token",
        "score": 1.0,
        "weighted_degree": 33.391252
      },
      {
        "id": 18,
        "label": "خواب",
        "type": "token",
        "score": 0.981621,
        "weighted_degree": 32.777547
      },
      {
        "id": 37,
        "label": "سو",
        "type": "token",
        "score": 0.981621,
        "weighted_degree": 32.777547
      },
      {
        "id": 41,
        "label": "کلیدی",
        "type": "token",
        "score": 0.958483,
        "weighted_degree": 32.004958
      },
      {
        "id": 12,
        "label": "دهان",
        "type": "token",
        "score": 0.934198,
        "weighted_degree": 31.194028
      },
      {
        "id": 27,
        "label": "کنار",
        "type": "token",
        "score": 0.934198,
        "weighted_degree": 31.194028
      },
      {
        "id": 38,
        "label": "خانه",
        "type": "token",
        "score": 0.926679,
        "weighted_degree": 30.942974
      },
      {
        "id": 22,
        "label": "پای",
        "type": "token",
        "score": 0.898587,
        "weighted_degree": 30.004958
      }
    ],
    "important_relations": [
      {
        "source": 12,
        "target": 13,
        "source_label": "دهان",
        "target_label": "پنجره",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 13,
        "target": 12,
        "source_label": "پنجره",
        "target_label": "دهان",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 14,
        "target": 15,
        "source_label": "باران",
        "target_label": "می‌ریخت",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 15,
        "target": 14,
        "source_label": "می‌ریخت",
        "target_label": "باران",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 18,
        "target": 19,
        "source_label": "خواب",
        "target_label": "خاکستر",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 19,
        "target": 18,
        "source_label": "خاکستر",
        "target_label": "خواب",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 19,
        "target": 20,
        "source_label": "خاکستر",
        "target_label": "می‌دید",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 20,
        "target": 19,
        "source_label": "می‌دید",
        "target_label": "خاکستر",
        "weight": 3.110434,
        "relation": "pmi"
      }
    ],
    "reasoning_path": [
      "دهان -[pmi]-> پنجره",
      "پنجره -[pmi]-> دهان",
      "باران -[pmi]-> می‌ریخت",
      "می‌ریخت -[pmi]-> باران",
      "خواب -[pmi]-> خاکستر"
    ],
    "graph_memory_report": {},
    "focus_terms": [],
    "focus_nodes": [],
    "focus_relations": [],
    "graph_metrics": {
      "node_count": 52,
      "edge_count": 532,
      "directed": false,
      "relation_edge_counts": {
        "cooccurrence": 166,
        "pmi": 166,
        "dependency": 86,
        "stem": 0,
        "subword": 0,
        "word_document": 90,
        "topic_document": 24
      },
      "node_type_counts": {
        "token": 45,
        "document": 1,
        "topic": 6
      },
      "enabled_relations": [
        "cooccurrence",
        "pmi",
        "dependency",
        "stem",
        "subword",
        "word_document",
        "topic_document"
      ]
    }
  },
  "explanation": {
    "status": "success",
    "task": "explainable_graph_nlp",
    "summary": "خروجی بر پایه گره‌های مرکزی و ارتباطات پرتکرار گراف توضیح داده شد.",
    "keywords": [
      {
        "text": "آینه",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "سایه",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "قدیمی",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "نامم",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "دهان",
        "score": 0.030303,
        "count": 1
      },
      {
        "text": "پنجره",
        "score": 0.030303,
        "count": 1
      }
    ],
    "top_nodes": [
      {
        "id": 23,
        "label": "گم‌شده‌ام",
        "type": "token",
        "score": 1.0,
        "weighted_degree": 33.391252
      },
      {
        "id": 18,
        "label": "خواب",
        "type": "token",
        "score": 0.981621,
        "weighted_degree": 32.777547
      },
      {
        "id": 37,
        "label": "سو",
        "type": "token",
        "score": 0.981621,
        "weighted_degree": 32.777547
      },
      {
        "id": 41,
        "label": "کلیدی",
        "type": "token",
        "score": 0.958483,
        "weighted_degree": 32.004958
      },
      {
        "id": 12,
        "label": "دهان",
        "type": "token",
        "score": 0.934198,
        "weighted_degree": 31.194028
      },
      {
        "id": 27,
        "label": "کنار",
        "type": "token",
        "score": 0.934198,
        "weighted_degree": 31.194028
      }
    ],
    "important_relations": [
      {
        "source": 12,
        "target": 13,
        "source_label": "دهان",
        "target_label": "پنجره",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 13,
        "target": 12,
        "source_label": "پنجره",
        "target_label": "دهان",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 14,
        "target": 15,
        "source_label": "باران",
        "target_label": "می‌ریخت",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 15,
        "target": 14,
        "source_label": "می‌ریخت",
        "target_label": "باران",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 18,
        "target": 19,
        "source_label": "خواب",
        "target_label": "خاکستر",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 19,
        "target": 18,
        "source_label": "خاکستر",
        "target_label": "خواب",
        "weight": 3.110434,
        "relation": "pmi"
      }
    ],
    "reasoning_path": [
      "دهان -[pmi]-> پنجره",
      "پنجره -[pmi]-> دهان",
      "باران -[pmi]-> می‌ریخت",
      "می‌ریخت -[pmi]-> باران",
      "خواب -[pmi]-> خاکستر",
      "خاکستر -[pmi]-> خواب"
    ],
    "graph_memory_report": {},
    "focus_terms": [],
    "focus_nodes": [],
    "focus_relations": [],
    "graph_metrics": {
      "node_count": 52,
      "edge_count": 532,
      "directed": false,
      "relation_edge_counts": {
        "cooccurrence": 166,
        "pmi": 166,
        "dependency": 86,
        "stem": 0,
        "subword": 0,
        "word_document": 90,
        "topic_document": 24
      },
      "node_type_counts": {
        "token": 45,
        "document": 1,
        "topic": 6
      },
      "enabled_relations": [
        "cooccurrence",
        "pmi",
        "dependency",
        "stem",
        "subword",
        "word_document",
        "topic_document"
      ]
    }
  },
  "graph_memory": {
    "status": "success",
    "task": "graph_memory_generation",
    "summary": "بر اساس حافظه گرافی، مفاهیم محوری عبارت‌اند از: آینه، سایه، قدیمی، نامم. شواهد مرتبط: در آینه، سایه‌ام از من قدیمی‌تر بود\nو نامم از دهانِ پنجره به باران می‌ریخت\n\nچراغی که خوابِ خاکستر می‌دید\nراه را به پای گم‌شده‌ام نشان نمی‌داد\n\nرود از کنار من گذشت\nاما تشنگی در مشت‌هایم لانه کرده بود\n\nگفتم: کدام سو خانه است باد، کلیدی زنگ‌زده را در سکوت چرخاند",
    "keywords": [
      {
        "text": "آینه",
        "score": 0.030303,
        "count": 2
      },
      {
        "text": "سایه",
        "score": 0.030303,
        "count": 2
      },
      {
        "text": "قدیمی",
        "score": 0.030303,
        "count": 2
      },
      {
        "text": "نامم",
        "score": 0.030303,
        "count": 2
      }
    ],
    "top_nodes": [
      {
        "id": 23,
        "label": "گم‌شده‌ام",
        "type": "token",
        "score": 1.0,
        "weighted_degree": 46.057919
      },
      {
        "id": 41,
        "label": "کلیدی",
        "type": "token",
        "score": 0.969901,
        "weighted_degree": 44.671625
      },
      {
        "id": 38,
        "label": "خانه",
        "type": "token",
        "score": 0.968555,
        "weighted_degree": 44.60964
      },
      {
        "id": 12,
        "label": "دهان",
        "type": "token",
        "score": 0.952294,
        "weighted_degree": 43.860695
      },
      {
        "id": 27,
        "label": "کنار",
        "type": "token",
        "score": 0.952294,
        "weighted_degree": 43.860694
      },
      {
        "id": 37,
        "label": "سو",
        "type": "token",
        "score": 0.943252,
        "weighted_degree": 43.444213
      },
      {
        "id": 18,
        "label": "خواب",
        "type": "token",
        "score": 0.943252,
        "weighted_degree": 43.444213
      },
      {
        "id": 22,
        "label": "پای",
        "type": "token",
        "score": 0.883054,
        "weighted_degree": 40.671625
      }
    ],
    "important_relations": [
      {
        "source": 12,
        "target": 13,
        "source_label": "دهان",
        "target_label": "پنجره",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 13,
        "target": 12,
        "source_label": "پنجره",
        "target_label": "دهان",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 14,
        "target": 15,
        "source_label": "باران",
        "target_label": "می‌ریخت",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 15,
        "target": 14,
        "source_label": "می‌ریخت",
        "target_label": "باران",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 18,
        "target": 19,
        "source_label": "خواب",
        "target_label": "خاکستر",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 19,
        "target": 18,
        "source_label": "خاکستر",
        "target_label": "خواب",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 19,
        "target": 20,
        "source_label": "خاکستر",
        "target_label": "می‌دید",
        "weight": 3.110434,
        "relation": "pmi"
      },
      {
        "source": 20,
        "target": 19,
        "source_label": "می‌دید",
        "target_label": "خاکستر",
        "weight": 3.110434,
        "relation": "pmi"
      }
    ],
    "reasoning_path": [
      "دهان -[pmi]-> پنجره",
      "پنجره -[pmi]-> دهان",
      "باران -[pmi]-> می‌ریخت",
      "می‌ریخت -[pmi]-> باران",
      "خواب -[pmi]-> خاکستر"
    ],
    "graph_memory_report": {
      "enabled": true,
      "seed_nodes": 45,
      "retrieved_nodes": 16,
      "retrieved_edges": 64,
      "top_nodes": [
        "نشان",
        "خانه",
        "باران",
        "گم‌شده‌ام",
        "سو",
        "پای",
        "پنجره"
      ],
      "relation_counts": {
        "pmi": 42,
        "cooccurrence": 12,
        "dependency": 10
      },
      "coverage": 0.3333333333333333,
      "prompt_concepts": [
        "آینه",
        "سایه‌ام",
        "قدیمی‌تر",
        "نامم",
        "دهان",
        "پنجره",
        "باران",
        "می‌ریخت",
        "چراغی",
        "خواب",
        "خاکستر",
        "می‌دید"
      ]
    },
    "focus_terms": [],
    "focus_nodes": [],
    "focus_relations": [],
    "graph_metrics": {
      "node_count": 53,
      "edge_count": 634,
      "directed": false,
      "relation_edge_counts": {
        "cooccurrence": 166,
        "pmi": 166,
        "dependency": 86,
        "stem": 0,
        "subword": 0,
        "word_document": 180,
        "topic_document": 36
      },
      "node_type_counts": {
        "token": 45,
        "document": 2,
        "topic": 6
      },
      "enabled_relations": [
        "cooccurrence",
        "pmi",
        "dependency",
        "stem",
        "subword",
        "word_document",
        "topic_document"
      ]
    }
  }
}
```

## Rakhshai MCP Evidence Summary

### Top Nodes

```text
گم‌شده‌ام، خواب، سو، کلیدی، دهان، کنار
```

### Important Relations

```text
دهان -[pmi]-> پنجره
پنجره -[pmi]-> دهان
باران -[pmi]-> می‌ریخت
می‌ریخت -[pmi]-> باران
خواب -[pmi]-> خاکستر
خاکستر -[pmi]-> خواب
```

### Reasoning Path

```text
دهان -[pmi]-> پنجره
پنجره -[pmi]-> دهان
باران -[pmi]-> می‌ریخت
می‌ریخت -[pmi]-> باران
خواب -[pmi]-> خاکستر
خاکستر -[pmi]-> خواب
```

### Graph Summary Top Nodes

```text
گم‌شده‌ام، خواب، سو، کلیدی، دهان، کنار، خانه، پای
```

### Graph Memory Report

```json
{
  "enabled": true,
  "seed_nodes": 45,
  "retrieved_nodes": 16,
  "retrieved_edges": 64,
  "top_nodes": [
    "نشان",
    "خانه",
    "باران",
    "گم‌شده‌ام",
    "سو",
    "پای",
    "پنجره"
  ],
  "relation_counts": {
    "pmi": 42,
    "cooccurrence": 12,
    "dependency": 10
  },
  "coverage": 0.3333333333333333,
  "prompt_concepts": [
    "آینه",
    "سایه‌ام",
    "قدیمی‌تر",
    "نامم",
    "دهان",
    "پنجره",
    "باران",
    "می‌ریخت",
    "چراغی",
    "خواب",
    "خاکستر",
    "می‌دید"
  ]
}
```

## Automatic Evidence Metrics

- direct_evidence_hits: `1`
- mcp_evidence_hits: `4`
- evidence_improvement_ratio: `4.0x`
- has_explanation_signal direct: `False`
- has_explanation_signal rakhshai_mcp: `True`

## Manual Scoring Rubric

Each output is scored from 1 to 5 on these criteria:

- interpretation_accuracy
- faithfulness_to_text
- symbol_relation_quality
- evidence_usage
- explanation_quality

Penalty:

- hallucination_penalty: `0` to `3`

Formula:

```text
final_score = interpretation_accuracy
  + faithfulness_to_text
  + symbol_relation_quality
  + evidence_usage
  + explanation_quality
  - hallucination_penalty
```

## Manual Scoring Result

### direct

```json
{
  "interpretation_accuracy": 5,
  "faithfulness_to_text": 5,
  "symbol_relation_quality": 4,
  "evidence_usage": 3,
  "explanation_quality": 4,
  "hallucination_penalty": 0
}
```
Final direct_score: `21`

### rakhshai_mcp

```json
{
  "interpretation_accuracy": 5,
  "faithfulness_to_text": 5,
  "symbol_relation_quality": 5,
  "evidence_usage": 5,
  "explanation_quality": 5,
  "hallucination_penalty": 0
}
```
Final mcp_score: `25`

### Improvement

- improvement_ratio: `1.1905x`
- improvement_percent: `19.0%`

Rakhshai MCP در این تست `1.1905x`، یعنی حدود `19.0%`، بهتر از حالت direct امتیاز گرفت.

## Success Criteria

- `mcp_score > direct_score`
- `improvement_ratio > 1.0`
- `mcp_evidence_hits > direct_evidence_hits`
- `has_explanation_signal` in MCP is better than direct

### Result

```json
{
  "direct_evidence_hits": 1,
  "mcp_evidence_hits": 4,
  "evidence_improvement_ratio": 4.0,
  "direct_has_explanation_signal": false,
  "mcp_has_explanation_signal": true,
  "checks": {
    "manual_mcp_score_gt_direct": true,
    "manual_improvement_ratio_gt_1": true,
    "mcp_evidence_hits_gt_direct": true,
    "mcp_has_better_explanation_signal": true
  },
  "success": true
}
```

در این تست تک‌نمونه‌ای، اتصال `gpt-5.4` به Rakhshai MCP باعث شد پاسخ مدل در تحلیل شعر فارسی دقیق‌تر، متن‌وفادارتر، مستندتر و توضیح‌پذیرتر شود.

بر اساس امتیاز داوری‌شده، Rakhshai MCP نسبت به direct برابر با `1.1905x` بهتر عمل کرد، یعنی حدود `19.0%` بهبود داشت.

در metric خودکار evidence، Rakhshai MCP نسبت به direct برابر با `4.0x` شواهد بیشتری را وارد پاسخ کرد.

## Response Received: direct

```text
این شعر به‌نظر می‌رسد درباره‌ی **گم‌گشتگی، بحران هویت، و نرسیدن به آرامش یا خانه‌ی واقعی** باشد. شاعر با چند تصویر نمادین نشان می‌دهد که آدمی گاهی خودش را گم می‌کند؛ اسمش، راهش، تشنگی‌اش و حتی معنای خانه برایش مبهم می‌شود.

### معنی ساده‌ی اصلی شعر
راوی حس می‌کند دیگر با خودش یکی نیست.
وقتی به آینه نگاه می‌کند، انگار چیزی در وجودش هست که از خودش هم قدیمی‌تر و عمیق‌تر است؛ یک سایه، یک گذشته، یا زخمی کهنه.
نامش هم دیگر ثابت و محکم نیست، بلکه انگار از دهان پنجره به باران می‌ریزد؛ یعنی هویت او در حال پخش شدن، محو شدن یا از دست رفتن است.

بعد از آن، چراغی که باید راه را نشان بدهد، خودش درگیر خواب خاکستر است؛ یعنی روشنایی و امید هم ضعیف شده و نمی‌تواند راه نجات را نشان دهد.
رود از کنارش می‌گذرد، یعنی امکان زندگی، حرکت، یا رهایی وجود دارد؛ اما تشنگی هنوز در مشت‌های او مانده، یعنی با اینکه چیزی برای رفع نیازش هست، او نمی‌تواند به آن برسد یا از آن بهره بگیرد.

در پایان، راوی می‌پرسد خانه کجاست؛ یعنی جای امن، آرامش، هویت، یا مقصد واقعی‌اش کجاست.
اما پاسخ روشنی نمی‌گیرد؛ فقط باد کلیدی زنگ‌زده را در سکوت می‌چرخاند. این تصویر یعنی شاید راهی برای رسیدن به خانه هست، اما این راه قدیمی، فراموش‌شده، سخت یا خاموش است.

---

## رابطه‌ی میان این نمادها

### ۱. آینه
آینه نماد **خودشناسی** و روبه‌رو شدن با خویشتن است.
اما در این شعر، آینه حقیقتی آرام و روشن نشان نمی‌دهد؛ بلکه سایه‌ای قدیمی‌تر را نشان می‌دهد. پس آینه اینجا محل کشف یک **خودِ زخمی، پنهان یا تاریخی** است.

### ۲. سایه
سایه می‌تواند نماد **بخش تاریک، پنهان، یا گذشته‌ی انسان** باشد.
اینکه سایه از «من» قدیمی‌تر است یعنی گذشته، ترس‌ها، رنج‌ها یا لایه‌های ناآگاه وجود انسان از هویت فعلی او ریشه‌دارتر شده‌اند.
پس سایه با آینه رابطه دارد: آینه قرار است «من» را نشان دهد، اما به‌جای منِ روشن، سایه را نشان می‌دهد.

### ۳. نام
نام نماد **هویت، شناخته‌شدن، و تعریف انسان از خودش** است.
وقتی نام از دهان پنجره به باران می‌ریزد، یعنی هویت راوی در حال پراکندگی و محو شدن است.
نام با آینه و سایه مرتبط است، چون هر سه به مسئله‌ی «من کیستم؟» مربوط‌اند.

### ۴. چراغ
چراغ نماد **روشنایی، آگاهی، امید و راهنمایی** است.
اما این چراغ خواب خاکستر می‌بیند؛ یعنی به‌جای نور و بیداری، به خاموشی و پایان نزدیک شده است.
پس چراغ باید راه را نشان دهد، اما نمی‌دهد. این یعنی راوی در بحران هویت فقط خودش را گم نکرده، بلکه **نشانه‌های هدایت** را هم از دست داده است.

### ۵. رود
رود نماد **زندگی، حرکت، گذر زمان، امکان رهایی یا پاکی** است.
رود از کنار راوی می‌گذرد، یعنی فرصت یا امکان نجات وجود دارد.
اما او همچنان تشنه است. پس رود با تشنگی رابطه‌ی مستقیم دارد: **آب هست، اما سیراب‌شدن نیست**.

### ۶. تشنگی
تشنگی نماد **نیاز عمیق روحی، کمبود معنا، عشق، آرامش یا حقیقت** است.
این تشنگی در مشت‌ها لانه کرده؛ یعنی راوی می‌خواهد چیزی را بگیرد، نگه دارد یا به دست آورد، اما دست‌هایش به‌جای آب، فقط تشنگی را نگه داشته‌اند.
این تصویر خیلی مهم است: یعنی مشکل فقط نبودن آب نیست، بلکه **ناتوانی در دریافت آن** است.

### ۷. خانه
خانه نماد **آرامش، تعلق، هویت، اصل خویش، یا مقصد نهایی** است.
وقتی راوی می‌پرسد «کدام سو خانه است؟» یعنی دیگر نمی‌داند به کجا تعلق دارد.
خانه در این شعر فقط یک مکان واقعی نیست؛ بیشتر یک **پناه معنوی و درونی** است.

### ۸. کلید
کلید نماد **راه‌حل، ورود، کشف، و رسیدن به خانه یا حقیقت** است.
اما کلید زنگ‌زده است؛ یعنی راه رسیدن به خانه وجود دارد، ولی کهنه، فراموش‌شده، دشوار یا مدت‌ها بی‌استفاده مانده است.
این کلید را باد در سکوت می‌چرخاند، نه خود راوی؛ یعنی پاسخ هنوز در اختیار کامل او نیست و بیشتر به شکل مبهم و غیرمستقیم ظاهر می‌شود.

---

## جمع‌بندی رابطه‌ی همه‌ی این عناصر
این عناصر یک زنجیره می‌سازند:

- **آینه، سایه و نام** → بحران هویت و ناآشنایی با خود
- **چراغ** → ناتوانی در پیدا کردن راه
- **رود و تشنگی** → وجود امکان نجات، اما نرسیدن به آن
- **خانه و کلید** → جست‌وجوی آرامش و حقیقت، با راهی که هست اما بسته یا فراموش شده

### در یک جمله:
شعر می‌گوید انسان گاهی خودش را گم می‌کند؛ هویتش محو می‌شود، راه را پیدا نمی‌کند، با وجود نزدیکیِ نجات همچنان تشنه می‌ماند، و در جست‌وجوی خانه‌ای است که کلیدش هست، اما زنگ‌زده و خاموش مانده است.

اگر بخواهی، می‌توانم همین پاسخ را هم به شکل **خیلی کوتاه و مناسبِ نوشتن در برگه‌ی امتحان** بازنویسی کنم.
```

## Response Received: rakhshai_mcp

```text
به نظر من معنی اصلی شعر این است که **گوینده دچار گم‌گشتگیِ درونی و بحران هویت است**: خودش را می‌بیند، اما آنچه در آینه می‌بیند با «خودِ زنده و حاضر» یکی نیست؛ راه هست، نشانه هست، حتی رود هم هست، اما او هنوز به خانه، آرامش، معنا یا حقیقت نمی‌رسد.

### معنی ساده‌ی کلی
شعر می‌گوید:
- من دیگر با خودم یکی نیستم.
- گذشته، سایه، و هویتِ فرسوده‌ام از خودِ اکنونم سنگین‌تر شده‌اند.
- اسم و هویتم هم انگار ثابت نیست و در جهان پخش می‌شود.
- چیزی که باید روشنایی بدهد، دیگر خاموش و بی‌اثر شده.
- با اینکه امکان نجات یا آرامش نزدیک است، درون من هنوز کمبود و تشنگی هست.
- در پایان، من دنبال «خانه» می‌گردم؛ یعنی جایی برای تعلق، آرامش، یا بازگشت به خود.
- اما پاسخ روشن نمی‌آید؛ فقط «کلیدی زنگ‌زده» در سکوت می‌چرخد، یعنی شاید راهی هست، اما کهنه، دشوار، مبهم و بی‌صدا.

## رابطه‌ی تصویرها با هم

### ۱) آینه و سایه
**آینه** جای دیدنِ خود است، اما در این شعر خودِ واقعی روشن نمی‌شود.
**سایه** معمولاً بخش تاریک، گذشته‌دار، یا نیمه‌پنهانِ انسان است.

وقتی می‌گوید:
**«در آینه، سایه‌ام از من قدیمی‌تر بود»**
یعنی گوینده حس می‌کند آن بخش تاریک یا زخمیِ وجودش از خودِ فعلی‌اش ریشه‌دارتر و مسلط‌تر است. انگار گذشته یا رنج، هویت او را ساخته است.

### ۲) نام
**نام** نشانه‌ی هویت است.
اما وقتی می‌گوید:
**«نامم از دهانِ پنجره به باران می‌ریخت»**
یعنی هویت او دیگر محکم و ثابت نیست؛ دارد از او بیرون می‌ریزد، در جهان حل می‌شود، یا پراکنده می‌شود.
پنجره مرز میان درون و بیرون است، و باران هم می‌تواند نشانه‌ی شسته‌شدن، پراکندگی، یا اندوه باشد.

### ۳) چراغ
**چراغ** معمولاً نماد آگاهی، امید، راهنمایی و کشف راه است.
اما اینجا:
**«چراغی که خوابِ خاکستر می‌دید / راه را ... نشان نمی‌داد»**
یعنی روشنایی هم دیگر کار نمی‌کند. چراغ به جای بیداری، «خواب» می‌بیند؛ آن هم خوابِ «خاکستر» ــ یعنی پایانِ آتش، خاموشی، فرسودگی.
پس امید یا آگاهیِ گوینده ضعیف شده و نمی‌تواند او را هدایت کند.

### ۴) رود و تشنگی
**رود** نماد زندگی، جریان، امکان، تازگی یا حقیقت است.
**تشنگی** نماد نیاز عمیق درونی، کمبود معنوی یا عاطفی است.

وقتی می‌گوید:
**«رود از کنار من گذشت / اما تشنگی در مشت‌هایم لانه کرده بود»**
یعنی حتی وقتی زندگی، فرصت، عشق یا معنا نزدیک اوست، باز هم نمی‌تواند آن را بگیرد.
«تشنگی در مشت‌هایم» خیلی مهم است: مشت هم حالت گرفتن دارد، هم حالت بسته‌بودن. شاید او آن‌قدر منقبض و درگیر است که حتی آبِ رود را هم نمی‌تواند بنوشد.

### ۵) خانه و کلید
**خانه** در این شعر فقط ساختمان نیست؛ بیشتر نمادِ آرامش، تعلق، اصلِ خویش، یا مقصد نهایی است.
**کلید** نماد گشودن، فهمیدن، ورود، و حل معماست.

اما:
**«باد، کلیدی زنگ‌زده را در سکوت چرخاند»**
یعنی پاسخ هست، اما روشن و زنده نیست.
کلید «زنگ‌زده» است: راهِ بازگشت یا رهایی قدیمی، فراموش‌شده، یا سخت‌کار شده.
اینکه **باد** آن را می‌چرخاند، نه خودِ گوینده، نشان می‌دهد اختیار کامل دست او نیست؛ نیروهای مبهم بیرونی یا سرنوشت دارند با این راز کار می‌کنند.
و «در سکوت» یعنی هنوز جواب صریحی داده نمی‌شود.

---

## جمع‌بندی رابطه‌ی همه‌ی این نمادها
این تصویرها یک زنجیره می‌سازند:

- **آینه**: تلاش برای دیدن خود
- **سایه**: خودِ تاریک‌تر و قدیمی‌تر
- **نام**: هویتِ در حال فروپاشی یا پراکندگی
- **چراغ**: ناتوانیِ آگاهی و امید در راهنمایی
- **رود**: امکان نجات یا معنا که از کنار می‌گذرد
- **تشنگی**: نیاز عمیقِ برطرف‌نشده
- **خانه**: مقصدِ آرامش و تعلق
- **کلید**: امکان رسیدن به آن مقصد، اما کهنه و خاموش

پس شعر درباره‌ی **انسانی است که هم خودش را گم کرده، هم راه را، هم خانه را؛ و با اینکه نشانه‌های نجات در جهان هست، هنوز درونش بسته و تشنه مانده است.**

## شواهد مرتبط از گراف Rakhshai MCP
فقط از بخش‌های مرتبط استفاده می‌کنم:

- **گره‌های مهم:**
  - **«گم‌شده‌ام»**: یکی از مرکزی‌ترین گره‌هاست و خیلی به خوانشِ «گم‌گشتگی هویتی و وجودی» کمک می‌کند.
  - **«خانه»** و **«کلیدی»**: برجسته بودن این دو گره نشان می‌دهد پایان شعر و مسئله‌ی رسیدن/گشودن، محور مهمی در معناست.
  - **«خواب»**: در کنار چراغ و خاکستر، به حس خاموشی و ناتوانیِ روشنایی کمک می‌کند.
  - **«سو»**: با پرسش «کدام سو خانه است؟» به مضمونِ جهت‌گم‌کردگی کمک می‌کند.

- **رابطه‌های مهم:**
  - **دهان ↔ پنجره**: این رابطه به تصویر عجیب و مهمِ «دهانِ پنجره» کمک می‌کند؛ یعنی پنجره مثل موجودی سخنگو یا مجرای بیرون‌ریختن هویت عمل می‌کند.
  - **باران ↔ می‌ریخت**: این رابطه خوانشِ «نامِ در حال پراکندگی و فروریختن» را تقویت می‌کند.
  - **خواب ↔ خاکستر**: این رابطه برای فهم «چراغی که خواب خاکستر می‌دید» خیلی مهم است و نشان می‌دهد چراغ به جای روشنایی، با خاموشی و پایان پیوند خورده است.

اگر بخواهم خیلی کوتاه در یک جمله بگویم:
**این شعر درباره‌ی آدمی است که از خودش دور افتاده، راه و خانه‌اش را گم کرده، و با اینکه نشانه‌های رسیدن وجود دارد، هنوز درونش بسته، خاموش و تشنه است.**
```

## How To Repeat

1. Put your API key in `.env.local` as `OPENAI_API_KEY=...`.
2. Install the OpenAI optional dependency:

```bash
pip install -e ".[openai]"
```

3. Run the benchmark and refresh this documentation page:

```bash
python scripts/evaluate_openai_mcp_persian.py \
  --model gpt-5.4 \
  --temperature 0 \
  --top-p 1 \
  --seed 42 \
  --max-output-tokens 3000 \
  --direct-manual-scores 5,5,4,3,4,0 \
  --mcp-manual-scores 5,5,5,5,5,0 \
  --report-path docs/mcp_single_poem_evaluation.md
```
