[English](README.md) | فارسی

# Rakhshai Graph-based NLP (RGN)

[![CI](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml)

**از دادهٔ خام فارسی تا گراف، مدل و محصول قابل استقرار.**

ما در [شرکت آریا هامان مهر پارسه](https://ariahaman.ir/)، `RGN` را به‌عنوان
نخستین سکوی یکپارچهٔ `NLP` گراف‌محور برای زبان فارسی و زیرساخت متن‌باز خود برای
ساخت گراف‌های چندرابطه‌ای و توسعهٔ مدل‌های زبانی گراف‌محور ایجاد کرده‌ایم.
رونمایی عمومی `RGN` در ۴ سپتامبر ۲۰۲۵ انجام شد؛ یعنی ۱۰ ماه و ۱۱ روز پیش از
رونمایی مدل هامان در ۱۵ ژوئیهٔ ۲۰۲۶ (۲۴ تیر ۱۴۰۵). `RGN` آماده‌سازی داده، ساخت
گراف، آموزش، ارزیابی،
استنتاج، مدیریت مدل، اتصال `MCP` و رابط کاربری راست‌به‌چپ را در یک مجموعهٔ
یکپارچهٔ `Python`، `CLI` و `UI` ارائه می‌کند.

با `RGN` می‌توانید مدل مقاله‌نویسی هامان را به کار بگیرید، یک `Graph-LM` فارسی
بومی را روی دادهٔ خود آموزش دهید یا محصولاتی برای طبقه‌بندی، خلاصه‌سازی،
توصیه‌گر محتوا و تحلیل معنایی گراف‌محور بسازید.

## مدل آمادهٔ استفاده: `Haman Persian Article Graph-LLM 125M`

در ادامهٔ این مسیر، در تاریخ ۲۴ تیر ۱۴۰۵، ما
[`Haman Persian Article Graph-LLM 125M`](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m)
را به‌عنوان نخستین خروجی پروژهٔ `NLP` شرکت آریا هامان مهر پارسه رونمایی
کردیم. این مدل نخستین مدل ایرانی است که با معماری ایرانی توسعه داده شده است تا
در ادامه بتوانیم چهارچوبی برای تمرکز روی زبان فارسی ایجاد کنیم.
این مدل با دریافت موضوع، مخاطب، لحن و کنترل بخش‌ها، مقالهٔ فارسی ساختاریافته
تولید می‌کند و یک
`Transformer` از نوع
`decoder-only` را با `GCN` واژگانی در سطح پیکره و
`context-gated graph-token fusion` ترکیب می‌کند.

| منبع | پیوند |
| --- | --- |
| کارت مدل و وزن‌ها | [صفحهٔ مدل](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m) |
| دادهٔ آموزش | [دیتاست آموزش مدل هامان](https://huggingface.co/datasets/aria-haman/haman-fa-wikipedia-articles-186k) |
| گردش‌کار قابل تکرار آموزش | [نوت‌بوک آموزش در گوگل کولب](https://colab.research.google.com/drive/1E50ISg1ANoW_rrFeRNBRcDn6C0cwfLyT?usp=sharing) |
| هستهٔ اجرا و کد منبع | [مخزن کد Rakhshai Graph-based NLP (RGN)](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP) |

مدل در مخزن مستقل `Hugging Face` نگهداری می‌شود تا کاربران وزن‌های نسخه‌بندی‌شده
را بدون افزودن چک‌پوینت‌های حجیم به مخزن کد دریافت کنند. فهرست فایل‌ها،
نمونه‌های تولید، نکات استقرار و مجوز مدل در کارت مدل آمده است.

### دریافت و اجرای مدل هامان

`RGN` را نصب کنید، نسخهٔ منتشرشدهٔ مدل را بگیرید و مقاله بسازید:

```bash
python -m pip install -e .
python -m pip install huggingface_hub
python -c 'from huggingface_hub import snapshot_download; snapshot_download(repo_id="aria-haman/haman-fa-article-graph-llm-125m", local_dir="models/haman-fa-article-graph-llm-125m")'

rgnn-cli article-generate \
  --model models/haman-fa-article-graph-llm-125m \
  --topic "آینده هوش مصنوعی در آموزش فارسی" \
  --audience "دانشجویان" \
  --tone "تحلیلی" \
  --sections 4 \
  --max-new-tokens 700 \
  --output-format markdown \
  --output-path haman-article.md
```

> محصول ما، `Rakhshai Graph-based NLP (RGN)`، تأییدیهٔ
> دانش‌بنیان را دریافت کرده است. جزئیات استعلام رسمی در پایین صفحه آمده است.

## سکوی محصول `RGN`

`RGN` چهار لایهٔ به‌هم‌پیوسته را در اختیار تیم‌های محصول قرار می‌دهد. هر لایه
مستقل قابل استفاده است و در کنار هم یک گردش‌کار کامل هوش مصنوعی فارسی می‌سازند.

| لایه | قابلیت ارائه‌شده |
| --- | --- |
| پردازش فارسی | نرمال‌سازی، توکن‌سازی، تحلیل زبانی و ویژگی‌های قابل استفادهٔ مجدد |
| هوشمندی گرافی | گراف‌های متنی، نحوی، معنایی، سندی و چندرابطه‌ای |
| موتور بومی مدل | `rakhshai_graph_nlp.lm` برای توکن‌ساز، `Graph-LM`، آموزش، ارزیابی، حافظهٔ گراف و استنتاج |
| گردش‌کارهای محصول | `rakhshai_graph_nlp.llm` برای محصولات بومی کاربردمحور، با شروع از تولید مقالهٔ ساختاریافتهٔ فارسی |

گردش‌کارهای سطح بالای `llm` بر موتور سطح پایین `lm` ساخته می‌شوند و جایگزین آن
نیستند. این جداسازی، هستهٔ مدل را برای کاربردهای مختلف قابل استفاده نگه می‌دارد
و هم‌زمان فرمان‌ها و خروجی‌های ساختاریافته و پایدار در اختیار تیم محصول می‌گذارد.

ما این سکوی متن‌باز را در تیم [RakhshAI](https://rakhshai.com/) شرکت آریا هامان
مهر پارسه توسعه می‌دهیم و کد آن را تحت مجوز MIT منتشر می‌کنیم.

## قابلیت‌های محصول

- **سکوی یکپارچهٔ `NLP` گراف‌محور فارسی:** مسیر متن خام تا پیش‌پردازش، ساخت
  گراف، آموزش، ارزیابی، استنتاج و ذخیرهٔ خط تولید را در یک محصول ارائه می‌کنیم.
- **مدل آمادهٔ مقاله‌نویسی فارسی:** مدل هامان را همراه با وزن‌ها، دیتاست،
  گردش‌کار آموزش و رابط تولید ساختاریافته در اختیار کاربران قرار داده‌ایم.
- **خط تولید بومی `LLM`:** در لایهٔ سطح بالای `rakhshai_graph_nlp.llm` می‌توانید
  داده را آماده و ممیزی کنید، چک‌پوینت را آموزش دهید، آزمون تفکیکی اجرا کنید و
  خروجی `Markdown` یا `JSON` بگیرید.
- **موتور قابل استفادهٔ مجدد `Graph-LM`:** توکن‌ساز فارسی‌آگاه، رمزگشای زبانی
  مدرن، رمزگذارهای `GCN`، `GraphSAGE`، `GAT` و `RGCN`، ترکیب تطبیقی گراف و
  متن، حافظهٔ گراف، ارزیابی و چک‌پوینت کامل در لایهٔ `lm` قرار دارند.
- **ابزارهای هوشمندی گرافی:** گراف‌های هم‌رخدادی، واژه-سند، شباهت اسناد،
  وابستگی نحوی، معنایی و چندرابطه‌ای می‌سازیم.
- **اجزای آمادهٔ کاربرد:** طبقه‌بندی، خلاصه‌سازی، توصیه‌گر محتوا، تشخیص
  نفرت‌پراکنی، تحلیل معنایی و تحلیل شبکه پشتیبانی می‌شوند.
- **رابط‌های متنوع محصول:** `API` پایدار پایتون، `rgnn-cli`، رابط وب
  راست‌به‌چپ و اتصال `MCP` برای عامل‌ها و سامانه‌های خودکار در دسترس‌اند.
- **کنترل‌های عملیاتی:** مسیرهای پشتیبانی‌شدهٔ `CPU` و `GPU`، مقایسهٔ حالت
  گرافی و بدون گراف، ذخیرهٔ کامل خروجی‌ها و کنترل تنظیمات تولید فراهم‌اند.

## اتصال MCP

ما قابلیت‌های تحلیل متن فارسی، ساخت گراف، خلاصه‌سازی گراف‌محور، بازیابی از
حافظهٔ گراف، تولید با `Graph-LM`، توضیح‌پذیری و دسترسی کنترل‌شده به منابع پروژه
را از طریق `MCP` ارائه می‌کنیم. تیم‌های محصول می‌توانند این ابزارها را به عامل،
`IDE`، چت‌بات و گردش‌کارهای خودکار متصل کنند.

- [راهنمای اتصال و استقرار MCP](docs/mcp.md)
- [گزارش منتشرشدهٔ ارزیابی MCP](docs/mcp_single_poem_evaluation.md)

## معماری `Graph-LM` در `RGN`

هستهٔ `Graph-LM` در `RGN` این مسیر را پیاده‌سازی می‌کند:

```text
Persian Text
→ PersianTokenizer
→ LM Dataset
→ Multi-Relation Persian Graph
→ RGN Graph Encoder (GCN / GraphSAGE / GAT / RGCN)
→ Adaptive Graph-Text Fusion
→ Low-Data Training Engine
→ Prompt-aware Graph Memory
→ Transformer Causal LM
→ Text Generation
```

اجزای اصلی این هسته:

```text
RGN Graph Encoder
+
Adaptive Graph-Text Fusion
+
Low-Data Training Engine
+
Persian Causal LM
```

به زبان ساده، `RGN` می‌تواند به جای یک مدل زبانی صرفاً دنباله‌ای، از رابطهٔ
گرافی واژه‌ها هم استفاده کند. مدل هنگام ساخت `embedding` نهایی هر توکن یاد
می‌گیرد چقدر به بازنمایی متنی و چقدر به بازنمایی گرافی اعتماد کند؛ یعنی ترکیب
به صورت ثابت و دستی نیست، بلکه با دروازهٔ قابل یادگیری انجام می‌شود. این
دروازه می‌تواند در سطح توکن، جمله و زیرگراف فعال شود.

> **سازگاری چک‌پوینت:** چک‌پوینت‌های قدیمی در حالت سازگاری بارگذاری می‌شوند؛
> برای استفاده از چینش فعلی رمزگشا و توکن‌ساز باید آموزش را تکرار کنید. جزئیات
> مهاجرت در [راهنمای `Graph-LM V2`](docs/graph_lm_v2.md) آمده است.

## مسیر مناسب محصول خود را انتخاب کنید

| هدف | مسیر پیشنهادی |
| --- | --- |
| تولید فوری مقالهٔ ساختاریافتهٔ فارسی | استفاده از [مدل منتشرشدهٔ هامان](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m) |
| ساخت یک `LLM` بومی برای کاربردی دیگر | شروع از [گردش‌کارهای سطح بالای `llm`](docs/llm.md) |
| آموزش یا توسعهٔ هستهٔ قابل استفادهٔ مجدد | استفاده از [موتور سطح پایین `lm`](docs/graph_lm_v2.md) |
| ساخت محصولات `NLP` گراف‌محور | استفاده از سازنده‌های گراف، مدل‌های `GNN`، وظایف آماده، `API` پایتون یا `CLI` همین مخزن |

## آموزش `Graph-LM` اختصاصی

برای شروع سریع محلی، یک نمونه از ویکی‌پدیای فارسی در مسیر زیر قرار دارد:

```text
data/wiki_fa_50k.txt
```

می‌توانید مستقیم با همین فایل آموزش و تولید متن را اجرا کنید. برای ساخت دوبارهٔ
فایل یا تهیهٔ نمونهٔ بزرگ‌تر، از اسکریپت زیر استفاده کنید:

```bash
python scripts/download_fa_wiki_sample.py \
  --output data/wiki_fa_50k.txt \
  --max-rows 50000 \
  --min-length 200
```

نمونه اجرای baseline بدون گراف:

```bash
rgnn-cli lm-train \
  --corpus data/wiki_fa_50k.txt \
  --graph-encoder none \
  --output-dir runs/wiki-baseline-lm
```

نمونه اجرای Graph-LM با GAT و fusion دروازه‌ای:

```bash
rgnn-cli lm-train \
  --corpus data/wiki_fa_50k.txt \
  --graph-encoder gat \
  --fusion gated \
  --output-dir runs/wiki-graph-lm
```

نمونه تولید متن از مدل Graph-LM:

```bash
rgnn-cli generate \
  --model runs/wiki-graph-lm \
  --prompt "امروز در تهران" \
  --max-new-tokens 100 \
  --temperature 0.8 \
  --top-k 50 \
  --repetition-penalty 1.2
```

مسیر مستقل آموزش engine-level برای LLM نیز بدون وابستگی به مدل بیرونی آماده
است:

```bash
rgnn-cli lm-build-corpus --input data/wiki_fa_test.txt --output-dir runs/fa-corpus
rgnn-cli lm-tokenize --corpus-dir runs/fa-corpus --output-dir runs/fa-shards
rgnn-cli lm-pretrain \
  --shard-manifest runs/fa-shards/shard_manifest.json \
  --output-dir runs/fa-pretrain \
  --model-profile tiny-test \
  --device cpu
rgnn-cli lm-ablation \
  --corpus runs/fa-corpus/train.txt \
  --output-dir runs/fa-ablation \
  --graph-encoders none gat \
  --device cpu
rgnn-cli lm-eval --model runs/fa-pretrain --eval-file runs/fa-corpus/validation.txt
rgnn-cli lm-run-report --run-dir runs/fa-pretrain
```

این مسیر از pretrained LM بیرونی، embedding آماده، distillation، داده synthetic
ساخته‌شده با LLM یا judge بیرونی استفاده نمی‌کند.

## گردش‌کار بومی مدل زبانی مقاله‌نویسی فارسی

این همان گردش‌کار محصولی است که
[مدل هامان](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m)
با آن ساخته شده است. می‌توانید چک‌پوینت آماده را همراه با
[دیتاست ۱۸۶ هزار مقاله‌ای](https://huggingface.co/datasets/aria-haman/haman-fa-wikipedia-articles-186k)
و [نوت‌بوک تمیز کولب](https://colab.research.google.com/drive/1E50ISg1ANoW_rrFeRNBRcDn6C0cwfLyT?usp=sharing)
استفاده کنید یا با فرمان‌های زیر، چک‌پوینت بومی و کاربردمحور خود را روی پیکرهٔ
فارسی اختصاصی آموزش دهید.

این گردش‌کار از طریق `rakhshai_graph_nlp.llm.article` ارائه می‌شود و توکن‌ساز،
سازندهٔ گراف، مدل، مربی و حافظهٔ گراف موتور سطح پایین
`rakhshai_graph_nlp.lm` را در یک خط تولید مقاله‌محور بسته‌بندی می‌کند؛ موتور
قابل استفادهٔ مجدد `Graph-LM` همچنان مستقل باقی می‌ماند.

جریان فنی:

- `article-prepare` دادهٔ خام مقاله را از TXT/JSONL/CSV/TSV نرمال می‌کند،
  قالب آموزشی مقاله‌محور می‌سازد و فایل‌های `corpus.txt`، `train.txt`،
  `validation.txt`، `prepared_articles.jsonl`، `rejected_records.jsonl` و
  `manifest.json` را می‌نویسد.
- `article-audit` کیفیت corpus بومی، خطر تکرار، شاخص‌های سطحی فارسی و رفتار
  tokenizer را قبل از آموزش سنگین بررسی می‌کند.
- `article-train` یک checkpoint مقاله‌محور Graph-LM را با artifactهای عادی
  مدل، tokenizer، گراف، graph memory و metrics آموزش می‌دهد و علاوه بر آن
  `article_llm_config.json` را برای metadata مربوط به workflow مقاله می‌نویسد.
- `article-ablation` variantهای بدون گراف، گراف‌دار، scopeهای گراف و گروه‌های
  relation را اجرا می‌کند و metrics اعتبارسنجی، fusion stats و zero-gate report
  را ثبت می‌کند.
- `article-generate` همان checkpoint را بارگذاری می‌کند و مقالهٔ فارسی
  ساختاریافته را به صورت Markdown یا JSON برمی‌گرداند. CLI همچنان دستورهای
  سطح بالای `article-*` را نگه می‌دارد و نام artifactها تغییر نمی‌کند.

مسیر کامل ساخت و آموزش:

1. دادهٔ مقاله را به صورت TXT، JSONL، CSV یا TSV آماده کنید. در فایل‌های
   ساختاریافته، فیلد `body` ضروری است و `title`، `summary`، `keywords` و
   `metadata` اختیاری هستند. برای رکوردهای شبیه ویکی‌پدیای فارسی می‌توانید از
   فیلدهای `title` و `text` همراه با `--training-format wikipedia_prompt`
   استفاده کنید.
2. با `article-prepare` رکوردها را نرمال کنید، متن‌های کوتاه را کنار بگذارید
   و splitهای تکرارپذیر `train.txt` و `validation.txt` بسازید.
3. قبل از آموزش سنگین، `article-audit` را اجرا کنید. این دستور پوشش نویسه‌های
   فارسی، خطر تکرار، پوشش metadata/source و رفتار tokenizer را گزارش می‌دهد؛
   برای انتخاب بین `word`، `bpe` و `unigram` می‌توانید probe آموزشی tokenizer
   را هم فعال کنید.
4. با `article-train` آموزش را اجرا کنید. نمونهٔ زیر از CUDA، AMP،
   `context_gated` graph fusion، tokenizer نوع Unigram و graph cache قابل‌استفاده
   مجدد استفاده می‌کند. برای ادامهٔ اجرای قطع‌شده، از
   `--resume-from runs/article-llm-fa` استفاده کنید.
5. بعد از آموزش، فایل‌های `metrics.json`، `article_llm_config.json`،
   `config.json`، `generation_config.json`، `tokenizer.json`، `model.pt`،
   `corpus.txt` و در اجراهای گراف‌دار `graph.pt` و `graph_memory.pt` را بررسی
   کنید.
6. checkpoint ساخته‌شده را با `article-generate` یا API پایتونی پایین استفاده
   کنید.

```bash
rgnn-cli article-prepare \
  --input data/persian_articles.jsonl \
  --output-dir runs/articles-prepared \
  --input-format jsonl \
  --min-body-chars 400 \
  --validation-ratio 0.1

rgnn-cli article-audit \
  --input data/persian_articles.jsonl \
  --output-dir runs/articles-audit \
  --input-format jsonl \
  --min-body-chars 400 \
  --tokenizer-types word bpe unigram

rgnn-cli article-train \
  --corpus runs/articles-prepared/corpus.txt \
  --output-dir runs/article-llm-fa \
  --device cuda \
  --amp \
  --batch-size 16 \
  --epochs 10 \
  --block-size 256 \
  --graph-encoder gat \
  --fusion context_gated \
  --graph-cache-dir runs/graph-cache \
  --tokenizer-type unigram \
  --unigram-num-pieces 32000
```

استفاده از checkpoint آموزش‌دیده از CLI:

```bash
rgnn-cli article-generate \
  --model runs/article-llm-fa \
  --topic "آینده هوش مصنوعی در آموزش فارسی" \
  --audience "دانشجویان" \
  --tone "تحلیلی" \
  --sections 4 \
  --max-new-tokens 700 \
  --output-format markdown \
  --output-path runs/article-llm-fa/education_article.md

rgnn-cli article-generate \
  --model runs/article-llm-fa \
  --topic "اقتصاد دیجیتال ایران" \
  --sections 4 \
  --output-format json \
  --output-path runs/article-llm-fa/economy_article.json
```

وقتی `article-train` همین `corpus.txt` آماده‌شده را می‌گیرد، از فایل‌های
کناری `train.txt` و `validation.txt` ساخته‌شده توسط `article-prepare` استفاده
می‌کند.

در `article-generate`، Graph Memory به صورت پیش‌فرض فعال است. اگر checkpoint
شامل `graph_memory.pt` باشد، همان حافظه بارگذاری می‌شود؛ اگر نباشد و
`corpus.txt` در checkpoint وجود داشته باشد، حافظه از corpus و
`graph_config.json` بازسازی می‌شود. برای خاموش‌کردن حافظه در تولید مقاله:

```bash
rgnn-cli article-generate \
  --model runs/article-llm-fa \
  --topic "آینده آموزش فارسی" \
  --graph-memory off
```

استفاده از همین checkpoint از پایتون:

```python
from rakhshai_graph_nlp.llm.article import (
    ArticleGenerationConfig,
    generate_persian_article,
)

article = generate_persian_article(
    ArticleGenerationConfig(
        model_dir="runs/article-llm-fa",
        topic="آینده آموزش فارسی",
        audience="دانشجویان",
        tone="تحلیلی",
        sections=4,
        max_new_tokens=700,
        graph_memory=True,
        device="cuda",
    )
)

print(article.full_markdown)
print(article.to_json())
```

## چرا گراف؟

متن فقط یک دنباله از کلمات نیست. در فارسی، رابطهٔ واژه‌ها، هم‌رخدادی‌ها،
وابستگی‌های نحوی، شباهت سندها و اتصال بین مفهوم‌ها اهمیت زیادی دارد. گراف
کمک می‌کند این رابطه‌ها را واضح‌تر ببینیم و به مدل بدهیم.

در رویکرد TextGCN، کلمات و سندها به گره‌های یک گراف تبدیل می‌شوند. رابطهٔ
کلمه با کلمه می‌تواند با PMI ساخته شود و رابطهٔ کلمه با سند با TF-IDF وزن
بگیرد. بعد مدل گرافی مثل GCN یا GAT اطلاعات را روی این شبکه پخش می‌کند و
برای گره‌های سند برچسب پیش‌بینی می‌کند.

به زبان ساده: به جای اینکه هر متن را جدا ببینیم، کل مجموعهٔ متن‌ها را مثل
یک شبکه می‌بینیم. این برای زبان فارسی جذاب است، چون معنی و نقش کلمات خیلی
وقت‌ها از رابطه‌شان با کلمات اطراف و سندهای دیگر روشن‌تر می‌شود.

## مؤلفه‌های اصلی

از نسخهٔ `2.2.0`، API پایتون پروژه پایدار است (`API_STATUS = "stable"` و
`__api_version__ = "2.2"`). برای کدهای کاربردی می‌توانید سطح رسمی را مستقیم
از `rakhshai_graph_nlp` یا از facade صریح `rakhshai_graph_nlp.api` import کنید:

```python
from rakhshai_graph_nlp import TextGraphClassifier, build_text_graph
from rakhshai_graph_nlp.api import GraphCausalLM, PersianTokenizer
```

راهنمای کامل و قدم‌به‌قدم استفاده از API در
[`docs/api_usage.md`](docs/api_usage.md) و مرجع پایدار API در
[`docs/api.md`](docs/api.md) قرار دارد.

| مؤلفه | توضیح |
| --- | --- |
| `Graph` | ظرف سبک گراف با adjacency، metadata گره‌ها، self-loop، درجه و نرمال‌سازی |
| `TextGraphClassifier` | کلاس اصلی برای آموزش، ارزیابی، پیش‌بینی، ذخیره و بارگذاری مدل طبقه‌بندی متن |
| `train_node_classifier` / `train_gcn_classifier` | helperهای سطح پایین‌تر برای آموزش طبقه‌بند گره با GNN |
| `build_text_graph` | ساخت گراف واژه-سند با PMI و TF-IDF برای TextGCN |
| `build_cooccurrence_graph` | ساخت گراف هم‌رخدادی واژه‌ها در پنجرهٔ لغتی |
| `build_document_graph` | ساخت گراف شباهت سندها با TF-IDF یا embedding |
| `build_dependency_graph` | ساخت گراف وابستگی نحوی با Stanza |
| `build_semantic_graph` | ساخت گراف معنایی از روابط واژگانی و شباهت embedding |
| `build_semantic_graph_from_farsnet` | ساخت گراف معنایی از خروجی JSON/CSV/TSV مربوط به FarsNet |
| `load_farsnet_relations` | خواندن روابط FarsNet از فایل و تبدیل آن به روابط قابل استفاده در گراف |
| `tokenize`، `tokenize_persian`، `PersianNormalizer` | helperهای پایدار برای توکن‌سازی و نرمال‌سازی فارسی |
| `graph_to_data`، `build_feature_matrix` | اتصال گراف‌های dense و ویژگی‌های متنی فارسی به PyTorch Geometric |
| `PersianTokenizer` | tokenizer عددی مخصوص LM با پشتیبانی از نیم‌فاصله، تمیزسازی فارسی، توکن مستقل برای علائم سجاوندی، نرمال‌سازی جداکننده‌های عدد و فولد قابل‌تنظیم همزه/اضافه، توکن ویژهٔ `<mask>`، و حالت‌های `word`/`char_chunk`/`bpe`/`unigram` (پیش‌فرض عملیاتی `unigram`) |
| `LMDataset` | آماده‌سازی `input_ids` و `target_ids` برای پیش‌بینی توکن بعدی |
| `build_graph_lm_graph` | ساخت گراف هم‌رخدادی واژگان از corpus برای Graph-LM |
| `GraphCausalLM` | مدل زبانی فارسی با GNN encoder، gated graph-token fusion و یک Transformer decoder مدرن (RoPE، SwiGLU، RMSNorm، تولید با KV-cache) |
| `RakhshaiGraphEncoder` | هستهٔ گرافی Graph-LM با پشتیبانی از `gcn`، `graphsage`، `gat`، `rgcn` و relation-aware encoding |
| `GraphMemoryArtifact` / `GraphMemoryConfig` | حافظهٔ گرافی مرتبط با prompt برای زمان تولید |
| `LMTrainer`، `LMTrainingConfig`، `train_graph_lm` | APIهای پایدار آموزش Graph-LM با checkpoint، validation و perplexity |
| `TextAugmentationConfig`، `augment_text`، `augment_graph_data` | ابزارهای regularization متن و گراف برای دادهٔ کم |
| `PoemRecommender`، `build_poem_index` | embedding، index و جست‌وجوی شعر با Graph-LM |
| `rakhshai_graph_nlp.llm.article` | namespace سطح workflow برای مقاله‌نویسی فارسی که روی موتور Graph-LM ساخته شده است |
| `ArticleCorpusConfig`، `prepare_article_corpus` | API آماده‌سازی دیتاست مقاله از ورودی‌های TXT/JSONL/CSV/TSV |
| `ArticleAuditConfig`، `audit_article_corpus` | API audit بومی corpus مقاله و benchmark توکنایزر |
| `ArticleTrainingConfig`، `train_article_llm` | پروفایل آموزش مقاله‌محور که همچنان artifactهای عادی checkpoint در Graph-LM را می‌نویسد |
| `ArticleAblationConfig`، `run_article_ablation` | runner بومی برای ablation گراف در workflow مقاله |
| `ArticleGenerationConfig`، `generate_persian_article`، `PersianArticle` | API تولید مقالهٔ فارسی ساختاریافته با خروجی Markdown و JSON |
| `--graph-encoder none` | baseline بدون گراف برای مقایسه با Graph-LM و سنجش اثر واقعی GNN/fusion |
| `GCNClassifier` | مدل GCN برای طبقه‌بندی گره‌ها |
| `GraphSAGEClassifier` | مدل GraphSAGE برای یادگیری از ساختار همسایگی |
| `GATClassifier` | مدل GAT با مکانیزم attention روی گراف |
| `textrank_summarise`، `gat_summarise` | APIهای خلاصه‌سازی استخراجی |
| `recommend_similar`، `HateSpeechDetector`، `compute_social_embeddings` | APIهای توصیه‌گر، تشخیص نفرت‌پراکنی و تحلیل شبکه |
| `accuracy`، `macro_f1`، `confusion_matrix` | metricهای پایدار ارزیابی |
| `rgnn-cli` | ابزار خط فرمان برای آموزش و ارزیابی سریع |

## وظایف تحلیلی

- **طبقه‌بندی متن فارسی:** دسته‌بندی خبر، پیام، نظر کاربر، محتوای شبکه
  اجتماعی یا هر متن برچسب‌دار دیگر.
- **خلاصه‌سازی استخراجی:** انتخاب جمله‌های مهم با `textrank_summarise` یا
  رتبه‌بند GATمحور `gat_summarise`.
- **توصیه‌گر محتوا:** پیدا کردن سندهای نزدیک به یک متن یا پرس‌وجو با
  `recommend_similar`.
- **تشخیص نفرت‌پراکنی:** کنترل سریع واژه‌محور با `contains_hate_speech` و
  مدل قابل آموزش با `HateSpeechDetector`.
- **تحلیل شبکه:** ساخت embedding برای گره‌ها با GraphSAGE و استفاده در
  خوشه‌بندی، تحلیل ارتباط‌ها یا تحلیل نفوذ.
- **گراف معنایی فارسی:** اتصال واژه‌های مرتبط با FarsNet، روابط دستی،
  synonymها یا embeddingهای فارسی.

## پشتیبانی از GPU

پروژه برای مدل‌های گرافی مبتنی بر PyTorch Geometric از GPU پشتیبانی می‌کند.
یعنی اگر سیستم شما کارت گرافیک NVIDIA، درایور مناسب و نسخه CUDA-compatible
از PyTorch داشته باشد، می‌توانید آموزش GCN، GraphSAGE و GAT را روی GPU اجرا
کنید.

مسیرهای GPU شامل این بخش‌ها هستند:

- `TextGraphClassifier(..., device="cuda")`
- `train_node_classifier(..., device="cuda")`
- `train_gcn_classifier(..., device="cuda")`
- اجرای CLI با `--device cuda`

نمونه استفاده در Python:

```python
from rakhshai_graph_nlp import TextGraphClassifier

clf = TextGraphClassifier(model="gcn", device="cuda", num_epochs=50)
clf.fit(texts, labels)
print(clf.predict(["تیم فوتبال امروز تمرین کرد"]))
```

نمونه استفاده در CLI:

```bash
rgnn-cli \
  --dataset benchmarks/persian_text_classification.csv \
  --model gcn \
  --device cuda \
  --epochs 50 \
  --output-dir runs/news-gcn-gpu
```

اگر CUDA در دسترس نباشد، مسیر CLI به CPU برمی‌گردد تا اجرا متوقف نشود. برای
بررسی CUDA در محیط خودتان:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

نکته مهم: الگوریتم‌هایی مثل TextRank، توصیه‌گر TF-IDF و بعضی ابزارهای NumPy
محور ذاتاً CPUمحور هستند. GPU بیشترین اثر را در آموزش مدل‌های GNN و اجرای
مدل‌های PyTorch/PyG نشان می‌دهد.

## نصب به زبان ساده

پیشنهاد می‌شود داخل یک virtual environment نصب کنید.

### macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

در بیشتر مک‌ها اجرای پروژه روی CPU انجام می‌شود. اگر روی مک Apple Silicon
کار می‌کنید، بخش‌های PyTorch ممکن است از شتاب‌دهنده‌های خود PyTorch استفاده
کنند، اما مسیر رسمی CLI این پروژه برای GPU فعلاً روی `cuda` متمرکز است.

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

برای GPU روی Linux، اول PyTorch سازگار با CUDA سیستم خودتان را نصب کنید و
بعد پروژه را نصب کنید:

```bash
python -c "import torch; print(torch.cuda.is_available())"
rgnn-cli --model gcn --device cuda
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ml]"
```

برای GPU روی Windows هم باید NVIDIA Driver و نسخه CUDA-compatible از PyTorch
درست نصب شده باشد:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
rgnn-cli --model gcn --device cuda
```

### افزودنی‌های اختیاری

```bash
python -m pip install -e ".[ml]"    # ابزارهای scikit-learn برای گراف سند و توصیه‌گر
python -m pip install -e ".[nlp]"   # پشتیبانی Stanza برای گراف وابستگی نحوی
python -m pip install -e ".[fa]"    # پشتیبانی Faiss
python -m pip install -e ".[docs]"  # ابزار ساخت مستندات
python -m pip install -e ".[dev]"   # ابزار تست، lint، build و انتشار
```

> **نکته مهم درباره دقت:** اگر از گراف وابستگی نحوی یا preset چندرابطه‌ای
> پیش‌فرض Graph-LM استفاده می‌کنید، نصب Stanza شدیداً توصیه می‌شود؛ چون این
> preset شامل رابطهٔ `dependency` است. اگر Stanza نصب نباشد، backend زبانی
> در حالت `auto` به backend ساده‌تر `heuristic` برمی‌گردد. پروژه همچنان اجرا
> می‌شود، اما کیفیت رابطه‌های `dependency`/lemma و در نتیجه دقت مدل‌هایی که به
> این رابطه‌های زبانی وابسته‌اند ممکن است پایین‌تر باشد.

### اجرای تست‌ها

```bash
python -m pip install pytest
python -m pytest
```

## 🖥️ رابط کاربری گرافیکی

ساده‌ترین راه برای دیدنِ تمام توان پروژه، یک رابط کاربری کامل، فارسی و
راست‌به‌چپ است که در فایل [`app.py`](app.py) قرار دارد.

### بالا آوردن رابط کاربری (آسان‌ترین راه)

```bash
# گزینهٔ ۱ — راه‌انداز یک‌دستوری (پیش‌نیازها را خودش نصب می‌کند)
./run_ui.sh

# گزینهٔ ۲ — اجرای مستقیم
pip install -r requirements-ui.txt   # یا: pip install -e ".[ui]"
python app.py
```

سپس مرورگر را روی نشانی **http://127.0.0.1:7860** باز کنید (راه‌انداز خودش
مرورگر را باز می‌کند).

### نصب پیش‌نیازها

راه‌انداز، پیش‌نیازهای **اصلی** و خودِ رابط کاربری را به‌طور خودکار نصب می‌کند.
علاوه بر این‌ها، پروژه چند **پکیج مکمل (اختیاری)** دارد که فقط در صورت نیاز به
قابلیت‌هایشان نصبشان کنید:

> ⚡ **برای استفاده از حداکثر توان پروژه، باید همهٔ پیش‌نیازها نصب باشند.** اگر
> می‌خواهید همهٔ قابلیت‌ها در دسترس باشند، همه را با یک دستور نصب کنید:
> `pip install -e ".[all]"`

| پکیج | نصب | چه چیزی را فعال می‌کند |
| --- | --- | --- |
| **scikit-learn** | `pip install "scikit-learn>=1.2"` | معیارها، خلاصه‌سازی TF-IDF، گراف سند |
| **stanza** | `pip install "stanza>=1.6"` | NLP پیشرفتهٔ فارسی: برچسب‌گذاری نقش، تکواژه‌یابی، تجزیهٔ نحوی |
| **faiss-cpu** | `pip install "faiss-cpu>=1.7.4"` | جست‌وجوی شباهتِ برداریِ سریع |

> **Stanza** برای بار اول به مدل زبانی فارسی‌اش هم نیاز دارد:
> `python -m stanza.download fa`

**نصب همه با یک دستور** (تمام پکیج‌های اختیاری):

```bash
pip install -e ".[all]"
```

**بررسی اینکه چه چیزهایی نصب است** — این دستور زیر هر پکیج می‌نویسد نصب هست یا نه
(و دستور نصب موارد نبوده را هم می‌دهد):

```bash
./run_ui.sh check
```

نمونهٔ خروجی:

```
Optional packages — install only the ones you need:

  • scikit-learn — ML metrics, TF-IDF summarization, document graphs
      ✅ installed
  • stanza — advanced Persian NLP: POS, lemma, dependency parsing
      ❌ not installed   →  python3 -m pip install "stanza>=1.6"
  • faiss-cpu — fast vector similarity search
      ❌ not installed   →  python3 -m pip install "faiss-cpu>=1.7.4"
```

### کنترل سرور

راه‌انداز `run_ui.sh` چند دستور ساده برای اجرا و توقف رابط کاربری دارد:

```bash
./run_ui.sh                 # اجرای رابط کاربری
./run_ui.sh --share         # اجرا + ساخت یک لینک عمومی موقت
./run_ui.sh --port 8080     # اجرا روی پورت دلخواه
./run_ui.sh stop            # توقف سرور و آزادسازی پورت
./run_ui.sh restart         # توقف و اجرای دوباره
./run_ui.sh status          # نمایش اینکه آیا در حال اجراست و روی کدام پورت
./run_ui.sh help            # نمایش همهٔ گزینه‌ها
```

- **برای توقف هنگام اجرا**، در همان ترمینال **`Ctrl + C`** بزنید؛ سرور تمیز بسته و
  پورت آزاد می‌شود. (از `Ctrl + Z` پرهیز کنید؛ آن فقط پروسه را **معلق** می‌کند و
  پورت اشغال می‌ماند.)
- **از یک ترمینال دیگر** هم می‌توانید هر زمان با `./run_ui.sh stop` آن را متوقف کنید.
- اگر پورت از اجرای قبلی اشغال مانده باشد، راه‌انداز پیش از شروع آن را **خودکار آزاد
  می‌کند**، تا هرگز با خطای «پورت در حال استفاده است» روبه‌رو نشوید.

> پیام‌های راه‌انداز در ترمینال انگلیسی‌اند، چون بیشتر ترمینال‌ها متن راست‌به‌چپ
> فارسی را درست نمایش نمی‌دهند. خودِ رابط وب کاملاً فارسی و راست‌به‌چپ است.

### چه چیزهایی در رابط کاربری هست؟

| برگه | کارکرد |
| --- | --- |
| ✂️ **توکنایزر فارسی** | توکن‌سازی با چهار الگوریتم (واژه/زیرواژه/BPE/Unigram)، نیم‌فاصله و تجزیهٔ تکواژی |
| 🕸️ **گراف متن** | ساخت و نمایش تصویریِ گراف چندرابطه‌ای با رنگ‌بندی روابط و انواع گره |
| 🎓 **آموزش Graph-LM** | آموزش مدل زبانی گرافی با **پیشرفت زندهٔ هر دوره** و نمودار سرگشتگی |
| ✨ **تولید متن** | تولید متن فارسی با **حافظهٔ گرافی** و نمایش گزارش بازیابی |
| 🗂️ **طبقه‌بندی متن** | آموزش و پیش‌بینی طبقه‌بند گرافی (GCN/GraphSAGE/GAT) |
| 🧰 **وظایف تحلیلی** | خلاصه‌سازی، پیشنهاد محتوا و تشخیص نفرت‌پراکنی |
| 🚀 **با تمام قدرت** | یک تور گام‌به‌گام که شما را تا حداکثر توان پروژه همراهی می‌کند |

### بخش «با تمام قدرت»

این برگه یک تور راهنماست: از یک پیکرهٔ خام شروع می‌کنید، گرافِ چندرابطه‌ایِ آن را
می‌بینید، یک Graph-LM کامل (با موتور آموزش کم‌داده و همهٔ زیان‌های چندوظیفه‌ای)
آموزش می‌دهید و در پایان با همان مدل و **حافظهٔ گرافی** متن فارسی تولید می‌کنید —
همه در چند کلیک.

## شروع سریع طبقه‌بندی گرافی

```python
from rakhshai_graph_nlp import TextGraphClassifier

texts = [
    "انتخابات و دولت و مجلس",
    "قانون و نمایندگان مجلس",
    "فوتبال و تیم ملی",
    "گل و مسابقه فوتبال",
]
labels = ["politics", "politics", "sports", "sports"]

clf = TextGraphClassifier(model="gcn", num_epochs=20)
clf.fit(texts, labels)

print(clf.evaluate(texts, labels))
print(clf.predict(["تیم فوتبال امروز تمرین کرد"]))

clf.save("runs/news-textgraph")

loaded = TextGraphClassifier.load("runs/news-textgraph")
print(loaded.predict(["مجلس درباره قانون جدید بحث کرد"]))
```

## رابط خط فرمان

ابزار `rgnn-cli` سه مسیر دارد: ابزارهای کلاسیک `NLP` گراف‌محور فارسی از طریق
فرمان پیش‌فرض طبقه‌بندی؛ موتور سطح پایین `lm` با فرمان‌هایی مانند
`lm-build-corpus`، `lm-pretrain`، `lm-train`، `lm-eval` و `generate`؛ و
گردش‌کارهای محصولی سطح بالای `llm` با `article-prepare`، `article-audit`،
`article-train`، `article-ablation` و `article-generate`. اگر زیرفرمان ندهید،
مسیر کلاسیک طبقه‌بندی اجرا می‌شود.

آموزش Graph-LM:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --fusion gated \
  --output-dir runs/graph-lm
```

در حالت پیش‌فرض، مسیر Graph-LM گراف چندرابطه‌ای کامل را می‌سازد:

```text
cooccurrence + pmi + dependency + stem + subword + word_document + topic_document
```

برای بازتولید baseline سادهٔ قدیمی، relation را صریح محدود کنید:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence \
  --output-dir runs/simple-graph-baseline
```

برای فعال‌کردن Graph Reasoning Core، می‌توانید relationها را به شکل
غنی‌تر به encoder بدهید. نمونهٔ سبک با GraphSAGE رابطه‌محور:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder graphsage \
  --graph-relation-mode embedding \
  --graph-pooling attention \
  --graph-node-importance \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --output-dir runs/graphsage-reasoning
```

نمونهٔ مستقیم با R-GCN:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-relation-mode rgcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --output-dir runs/rgcn-reasoning
```

برای فعال‌کردن Adaptive Graph-Text Fusion، سطح‌های fusion و شدت مصرف گراف را
صریح تنظیم کنید:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --fusion context_gated \
  --fusion-levels token,sentence,subgraph \
  --graph-fusion-scale 0.75 \
  --graph-fusion-dropout 0.1 \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --output-dir runs/adaptive-fusion
```

آموزش baseline بدون گراف برای مقایسه:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder none \
  --output-dir runs/baseline-lm
```

آموزش چندوظیفه‌ای Graph-LM به صورت پیش‌فرض فعال است. در هر batch
علاوه بر `next_token`، سیگنال‌های `masked_token`، `edge`، `neighbor`،
`node_relation`، `graph_text` و `sentence_graph` هم در صورت وجود داده لازم
محاسبه می‌شوند. اگر مدل را بدون گراف اجرا کنید، lossهای گرافی خودکار skip
می‌شوند و baseline بدون گراف همچنان قابل مقایسه می‌ماند.

نمونه اجرای کنترل‌شده با وزن‌دهی lossهای چندوظیفه‌ای:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --task-losses next_token,masked_token,edge,node_relation,graph_text,sentence_graph \
  --next-token-weight 1.0 \
  --masked-token-weight 0.25 \
  --edge-prediction-weight 0.1 \
  --node-relation-weight 0.1 \
  --graph-text-alignment-weight 0.1 \
  --sentence-graph-alignment-weight 0.1 \
  --mask-probability 0.15 \
  --negative-samples 1 \
  --output-dir runs/multitask-graph-lm
```

**Low-Data Training Engine** به صورت پیش‌فرض فعال است. این بخش
برای corpusهای کوچک طراحی شده و با augmentation متنی، dropout گرافی،
subgraph sampling، contrastive consistency، curriculum learning و early
stopping تلاش می‌کند مدل کمتر متن را حفظ کند و validation بهتری بگیرد.

نمونه اجرای صریح با تنظیمات پیشنهادی پیش‌فرض:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --augmentation-ratio 0.5 \
  --token-dropout 0.05 \
  --punctuation-dropout 0.5 \
  --node-dropout 0.05 \
  --edge-dropout 0.1 \
  --subgraph-sampling-ratio 0.9 \
  --contrastive-weight 0.05 \
  --early-stopping-patience 3 \
  --output-dir runs/low-data-training-engine
```

برای ablation یا برگشت به آموزش ساده‌تر، این قابلیت‌ها را می‌توانید خاموش کنید:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --no-text-augmentation \
  --edge-dropout 0 \
  --node-dropout 0 \
  --subgraph-sampling-ratio 1 \
  --contrastive-weight 0 \
  --no-curriculum \
  --early-stopping-patience 0 \
  --output-dir runs/no-low-data-regularization
```

تولید متن با checkpoint ذخیره‌شده:

```bash
rgnn-cli generate \
  --model runs/graph-lm \
  --prompt "امروز در تهران" \
  --max-new-tokens 100 \
  --min-new-tokens 20 \
  --temperature 0.8 \
  --top-k 50 \
  --repetition-penalty 1.2 \
  --graph-memory on \
  --graph-memory-top-k 32 \
  --graph-memory-depth 1 \
  --graph-memory-report-path runs/graph-lm/memory-report.json
```

وقتی `--graph-memory on` فعال باشد، generate ابتدا tokenهای prompt را به
nodeهای حافظه وصل می‌کند، همسایه‌های مرتبط را با وزن relationها امتیاز می‌دهد،
زیرگرافی محدود می‌سازد و همان زیرگراف را به Graph-LM می‌دهد. این حالت کمک
می‌کند اطلاعات نامرتبط از کل گراف کمتر وارد generation شود. وقتی
`--graph-memory off` باشد، retrieval انجام نمی‌شود و مدل مثل قبل از `graph.pt`
ثابت، گراف پویا یا حالت بدون گراف استفاده می‌کند.

مسیر Graph-LM برای corpus بزرگ‌تر هم آماده‌تر شده است. هدف این قابلیت‌ها
این است که ساخت گراف، ذخیره گراف، train و ادامه آموزش فقط برای corpusهای کوچک
مناسب نباشد. قابلیت‌های مقیاس‌پذیری شامل ساخت batchی آمار گراف، cache قابل
استفاده مجدد برای گراف، گزارش متریک‌های مقیاس‌پذیری، تنظیم DataLoader، AMP
اختیاری برای CUDA و resume کامل‌تر از checkpoint آموزشی است.

نمونه اجرای مقیاس‌پذیر برای corpus متوسط یا بزرگ:

```bash
rgnn-cli lm-train \
  --corpus data/wiki_fa_50k.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --graph-top-k 8 \
  --graph-build-batch-size 1000 \
  --graph-cache-dir runs/graph-cache \
  --dataloader-num-workers 2 \
  --dataloader-pin-memory \
  --amp \
  --output-dir runs/scalable-graph-lm
```

اگر آموزش قطع شد، می‌توانید از checkpoint آموزشی ادامه بدهید:

```bash
rgnn-cli lm-train \
  --corpus data/wiki_fa_50k.txt \
  --graph-encoder gcn \
  --graph-cache-dir runs/graph-cache \
  --resume-from runs/scalable-graph-lm \
  --epochs 10 \
  --output-dir runs/scalable-graph-lm
```

مقایسه رفتار قابلیت‌های مقیاس‌پذیری در حالت فعال و غیرفعال:

| قابلیت | وقتی فعال باشد | وقتی غیرفعال باشد |
| --- | --- | --- |
| `--graph-build-batch-size` | آمار هم‌رخدادی گراف در batchهای کوچک‌تر merge می‌شود و برای corpus بزرگ فشار حافظه کنترل‌پذیرتر است. | کل واحدهای متنی با مسیر قبلی پردازش می‌شوند؛ برای corpus کوچک ساده‌تر و کافی است. |
| `--graph-cache-dir` | گراف ساخته‌شده با hash وابسته به corpus، tokenizer و graph config ذخیره می‌شود؛ run بعدی با همان تنظیمات سریع‌تر از cache می‌خواند. | هر train گراف را از ابتدا می‌سازد؛ برای ablationهای سریع کوچک قابل قبول است، اما روی corpus بزرگ زمان را تکراراً مصرف می‌کند. |
| `--no-reuse-graph-cache` | اگر همراه cache استفاده شود، cache قبلی نادیده گرفته می‌شود و artifact تازه ساخته و جایگزین می‌شود. | حالت پیش‌فرض این است که اگر cache سازگار پیدا شود، همان استفاده شود. |
| `--graph-top-k` | تعداد همسایه‌های هر node محدود می‌شود؛ گراف خلوت‌تر، RAM کمتر و train سریع‌تر می‌شود، با احتمال حذف بعضی رابطه‌های ضعیف. | یال‌های بیشتری حفظ می‌شوند؛ سیگنال گرافی کامل‌تر است ولی هزینه حافظه و محاسبه بالاتر می‌رود. |
| خاموش‌کردن relationهای سنگین مثل `semantic_similarity` | ساخت گراف برای vocab بزرگ سریع‌تر و کم‌مصرف‌تر می‌شود. | semantic relation ممکن است کیفیت را بهتر کند، اما روی vocab بزرگ چون مقایسه جفتی دارد گران‌تر است. |
| `--dataloader-num-workers` | آماده‌سازی batchها می‌تواند موازی‌تر شود و GPU کمتر منتظر DataLoader بماند. | DataLoader تک‌پردازه می‌ماند؛ برای corpus کوچک ساده و کم‌هزینه است. |
| `--dataloader-pin-memory` | وقتی device واقعاً CUDA باشد، انتقال batchها به GPU روان‌تر می‌شود. | انتقال حافظه عادی انجام می‌شود؛ روی CPU/MPS اثر خاصی ندارد. |
| `--amp` | روی CUDA از automatic mixed precision استفاده می‌شود و معمولاً حافظه GPU و زمان train کمتر می‌شود. | آموزش با precision معمول PyTorch انجام می‌شود؛ پایدارترین مسیر برای CPU و debug است. |
| `--resume-from` | `model.pt`، optimizer state، epoch، best validation و RNG state از `training_state.pt` خوانده می‌شود و train از epoch بعدی ادامه می‌یابد. | آموزش از ابتدا شروع می‌شود، حتی اگر در output directory checkpoint قبلی وجود داشته باشد. |
| متریک‌های `graph_scalability` | `metrics.json` نشان می‌دهد cache فعال بوده یا نه، cache hit شده یا نه، ساخت گراف چقدر طول کشیده، چند batch گراف ساخته شده و graph چند node/edge دارد. | بدون این گزارش‌ها هم train کار می‌کند، اما تحلیل هزینه ساخت گراف و cache کمتر شفاف است. |

خروجی checkpoint مدل زبانی کامل است و فقط به `model.pt` محدود نمی‌شود:

```text
runs/graph-lm/
├── model.pt
├── training_state.pt
├── config.json
├── tokenizer.json
├── graph_config.json
├── graph.pt
├── graph_memory.pt
├── graph_memory_config.json
├── generation_config.json
├── metrics.json
└── corpus.txt
```

گزینه‌های مهم مسیر Graph-LM:

| گزینه | کاربرد |
| --- | --- |
| `--corpus` | مسیر فایل متن خام فارسی برای آموزش LM |
| `--tokenizer-type` | `word`، `char_chunk`، `bpe`، یا `unigram` (پیش‌فرض)؛ `unigram` کمترین OOV فارسی را دارد |
| `--unigram-num-pieces` | اندازهٔ هدفِ واژگانِ subword برای توکنایزر unigram (پیش‌فرض `8000`) |
| `--graph-encoder` | انتخاب `gcn`، `gat`، `graphsage`، `rgcn` یا `none` برای baseline بدون گراف |
| `--graph-relations` | انتخاب رابطه‌های گراف؛ اگر ندهید preset چندرابطه‌ای پیش‌فرض (که حالا شامل `dependency` است) فعال است |
| `--semantic-method` | روش `semantic_similarity`: `distributional` (PPMI-cosine، پیش‌فرض) یا `orthographic` (n-gram حروف) |
| `--linguistic-backend` | backend رابطه‌های `dependency`/lemma: `auto` (در صورت نصب Stanza، وگرنه heuristic)، `stanza`، یا `heuristic` |
| `--relation-weights` | وزن‌دهی relationها، مثل `pmi=0.7,stem=0.5` |
| `--graph-relation-mode` | نحوهٔ مصرف `edge_type` در encoder؛ یکی از `bias`، `embedding` (پیش‌فرض) یا `rgcn` |
| `--graph-pooling` | pooling اختیاری روی subgraphها؛ یکی از `none`، `mean` یا `attention` |
| `--graph-node-importance` | فعال‌کردن scorer داخلی برای تشخیص nodeهای مهم‌تر |
| `--no-graph-node-type-embedding` | خاموش‌کردن embedding نوع-نود (nodeهای غیرتوکنی از صفر شروع می‌کنند) |
| `--fusion` | انتخاب روش ترکیب embedding متنی و گرافی، مثل `gated` |
| `--fusion-levels` | انتخاب سطح‌های fusion؛ مثل `token` یا `token,sentence,subgraph` |
| `--graph-fusion-scale` | ضریب شدت embedding گرافی قبل از fusion |
| `--graph-fusion-dropout` | dropout روی embedding گرافی برای کاهش وابستگی بیش از حد به گراف |
| `--task-losses` | انتخاب lossهای چندوظیفه‌ای؛ پیش‌فرض همه taskهای چندوظیفه‌ای فعال‌اند |
| `--next-token-weight` و `--masked-token-weight` | وزن loss زبانی causal و masked-token |
| `--edge-prediction-weight` و `--neighbor-prediction-weight` | وزن پیش‌بینی یال و همسایه در گراف |
| `--node-relation-weight` | وزن تشخیص نوع رابطه از روی `edge_type` |
| `--graph-text-alignment-weight` و `--sentence-graph-alignment-weight` | وزن alignment بین نمایش متن و گراف |
| `--mask-probability` | احتمال انتخاب tokenهای غیر padding برای masked-token prediction |
| `--negative-samples` | تعداد نمونه منفی برای هر یال مثبت در lossهای گرافی |
| `--augmentation-ratio` | نسبت نمونه‌های augmented اضافه‌شده به train split؛ پیش‌فرض Low-Data Training Engine فعال است |
| `--token-dropout` و `--punctuation-dropout` | شدت augmentation متنی برای corpus کم‌داده |
| `--edge-dropout` و `--node-dropout` | regularization گرافی هنگام train برای جلوگیری از حفظ ساختار ثابت |
| `--subgraph-sampling-ratio` | سهم یال‌های نگه‌داشته‌شده در هر view گرافی train |
| `--contrastive-weight` | وزن loss سازگاری contrastive بین viewهای گرافی |
| `--no-text-augmentation` و `--no-curriculum` | خاموش‌کردن augmentation متنی یا curriculum برای ablation |
| `--checkpoint-metric` | سیگنال انتخاب best-checkpoint و early stopping: `next_token` (perplexity، پیش‌فرض) یا `total` (loss چندوظیفه‌ای) |
| `--early-stopping-patience` و `--early-stopping-min-delta` | کنترل early stopping بر اساس سیگنال `--checkpoint-metric` |
| `--max-grad-norm` | حد clipping گرادیان در trainer |
| `--graph-build-batch-size` | ساخت batchی آمار گراف برای corpusهای بزرگ‌تر |
| `--graph-cache-dir` | مسیر cache گراف‌های ساخته‌شده برای استفاده مجدد در runهای بعدی |
| `--no-reuse-graph-cache` | اجبار به بازسازی گراف حتی اگر cache سازگار وجود داشته باشد |
| `--dataloader-num-workers` | تعداد workerهای DataLoader در آموزش Graph-LM |
| `--dataloader-pin-memory` | فعال‌کردن pinned memory مؤثر هنگام train روی CUDA |
| `--amp` | فعال‌کردن mixed precision روی CUDA برای کاهش مصرف حافظه و افزایش سرعت |
| `--resume-from` | ادامه آموزش از checkpoint شامل model و training state |
| `--output-dir` | مسیر ذخیره checkpoint، configها، tokenizer و گزارش‌ها |
| `--temperature` | کنترل تصادفی‌بودن تولید متن؛ مقدار کمتر خروجی محافظه‌کارتر می‌دهد |
| `--top-k` | محدودکردن نمونه‌گیری به k توکن محتمل‌تر |
| `--min-new-tokens` | حداقل تعداد توکن جدید در تولید متن |
| `--repetition-penalty` | کاهش تکرار توکن‌ها در خروجی generate |
| `--graph-memory` | کنترل حافظه گرافی در generate؛ پیش‌فرض `on` است و با `off` خاموش می‌شود |
| `--graph-memory-top-k` | حداکثر تعداد nodeهای حافظه که برای prompt بازیابی می‌شوند |
| `--graph-memory-depth` | عمق گسترش همسایه‌ها از nodeهای prompt در حافظه |
| `--graph-memory-max-edges` | سقف یال‌های زیرگراف بازیابی‌شده برای کنترل هزینه و نویز |
| `--graph-memory-min-score` | حداقل امتیاز node برای ورود به زیرگراف حافظه |
| `--graph-memory-relation-weights` | وزن‌دهی relationها در retrieval، مثل `pmi=0.5,word_document=1.2` |
| `--graph-memory-report-path` | ذخیره گزارش JSON از nodeها، relationها و coverage حافظه بازیابی‌شده |

در مسیر طبقه‌بندی، اگر `--dataset` ندهید، یک آزمایش داخلی کوچک اجرا می‌شود
تا نصب و مدل پایه را smoke test کنید. اگر دیتاست بدهید، CLI متن‌ها را
می‌خواند، گراف واژه-سند می‌سازد، یکی از مدل‌های `gcn`، `graphsage` یا `gat`
را آموزش می‌دهد و گزارش train/validation/test می‌نویسد.

اجرای آزمایش داخلی کوچک:

```bash
rgnn-cli --model gcn --device cpu
```

اجرای pipeline روی دیتاست دارای ستون‌های `text` و `label`:

```bash
rgnn-cli \
  --dataset benchmarks/persian_text_classification.csv \
  --text-column text \
  --label-column label \
  --model gcn \
  --epochs 50 \
  --device cpu \
  --output-dir runs/news-gcn \
  --save-model runs/news-gcn/model.pt
```

اجرای همان مسیر روی GPU:

```bash
rgnn-cli \
  --dataset benchmarks/persian_text_classification.csv \
  --model gat \
  --device cuda \
  --epochs 50 \
  --output-dir runs/news-gat-gpu
```

ورودی دیتاست می‌تواند CSV، TSV یا JSONL باشد. به صورت پیش‌فرض ستون‌های
`text` و `label` خوانده می‌شوند، اما می‌توانید نام ستون‌ها را تغییر دهید:

```bash
rgnn-cli \
  --dataset data/comments.jsonl \
  --dataset-format jsonl \
  --text-column comment \
  --label-column sentiment \
  --model graphsage \
  --output-dir runs/comments-graphsage
```

همه گزینه‌ها را می‌توانید از فایل JSON هم بدهید؛ کلیدهای فایل همان نام
گزینه‌ها بدون `--` و با underscore هستند:

```json
{
  "dataset": "benchmarks/persian_text_classification.csv",
  "model": "gat",
  "epochs": 50,
  "hidden_dim": 16,
  "learning_rate": 0.005,
  "dropout": 0.3,
  "device": "cpu",
  "output_dir": "runs/news-gat"
}
```

```bash
rgnn-cli --config config.json
```

خروجی اجرا به طور پیش‌فرض در `OUTPUT_DIR/metrics.json` ذخیره می‌شود و شامل
نام دیتاست، مدل، دستگاه اجرا، نگاشت برچسب‌ها، اندازه splitها، accuracy و
macro-F1 برای train/validation/test و loss نهایی است. با `--report-path`
می‌توانید مسیر گزارش را دقیق تعیین کنید. با `--save-model` هم checkpoint
PyTorch مدل، نوع مدل، ابعاد، نگاشت برچسب‌ها و متریک‌ها ذخیره می‌شود؛ برای
pipeline سطح بالا و ذخیره/بارگذاری کامل‌تر، از `TextGraphClassifier.save`
در API پایتون استفاده کنید.

گزینه‌های مهم CLI:

| گزینه | کاربرد |
| --- | --- |
| `--dataset` | اجرای مسیر واقعی آموزش/ارزیابی روی CSV، TSV یا JSONL |
| `--dataset-format` | تعیین فرمت ورودی؛ مقدار `auto` از پسوند فایل تشخیص می‌دهد |
| `--text-column` و `--label-column` | نام فیلد متن و برچسب در دیتاست |
| `--train-ratio`، `--val-ratio`، `--test-ratio` | نسبت splitهای آموزش، اعتبارسنجی و آزمون |
| `--model` | انتخاب معماری بین `gcn`، `graphsage` و `gat` |
| `--epochs`، `--hidden-dim`، `--learning-rate`، `--weight-decay`، `--dropout` | تنظیمات آموزش مدل |
| `--gat-heads` | تعداد attention headها برای مدل `gat` |
| `--window-size` و `--min-count` | کنترل ساخت گراف متن با پنجره هم‌رخدادی و حداقل فراوانی واژه |
| `--device` | اجرای CPU یا CUDA؛ اگر CUDA در دسترس نباشد، CLI به CPU برمی‌گردد |
| `--output-dir` و `--report-path` | مسیر ذخیره گزارش اجرا |
| `--save-model` | ذخیره checkpoint مدل آموزش‌دیده |
| `--config` | خواندن همین گزینه‌ها از فایل JSON |
| `--log-level` و `--log-to` | کنترل log و اتصال اختیاری به `wandb` یا `mlflow` در آزمایش داخلی |

برای دیدن فهرست کامل گزینه‌ها:

```bash
rgnn-cli --help
```

## مثال‌های کاربردی

### خلاصه‌سازی

```python
from rakhshai_graph_nlp.tasks.summarization import (
    gat_summarise,
    textrank_summarise,
)

text = (
    "هوش مصنوعی در سال‌های اخیر رشد زیادی داشته است. "
    "بسیاری از شرکت‌ها از مدل‌های زبانی برای تحلیل متن استفاده می‌کنند. "
    "در زبان فارسی هم ابزارهای NLP در حال بهتر شدن هستند."
)

print(textrank_summarise(text, top_k=2))
print(gat_summarise(text, top_k=2))
```

### توصیه‌گر محتوا

```python
from rakhshai_graph_nlp.tasks.recommendation import recommend_similar

query = "این نمایشگاه جذاب است."
docs = [
    "این یک خبر سیاسی است و درباره انتخابات صحبت می‌کند.",
    "تیم فوتبال امروز بازی مهمی دارد.",
    "نمایشگاه جدید هنری با آثار نقاشان جوان افتتاح شد.",
]

print(recommend_similar(query, docs, top_k=2))
```

### تشخیص نفرت‌پراکنی

```python
from rakhshai_graph_nlp.tasks.hate_speech import (
    HateSpeechDetector,
    contains_hate_speech,
)

hate_terms = ["نفرت", "لعنت"]
print(contains_hate_speech("این پیام حاوی نفرت است", hate_terms))

texts = [
    "متن آرام و محترمانه",
    "گفتگوی خوب و عادی",
    "توهین بد و نفرت",
    "پیام بد و سمی",
]
labels = [False, False, True, True]

detector = HateSpeechDetector(num_epochs=20)
detector.fit(texts, labels)
print(detector.predict(["پیام حاوی نفرت"]))
detector.save("runs/hate-detector")
```

### گراف معنایی

```python
from rakhshai_graph_nlp.graphs.semantic import build_semantic_graph

words = ["گربه", "سگ", "ماشین"]
relations = {"گربه": ["سگ"]}
embeddings = {
    "گربه": [1.0, 0.0],
    "سگ": [0.9, 0.1],
    "ماشین": [0.0, 1.0],
}

graph = build_semantic_graph(
    words,
    relations=relations,
    embedding_lookup=embeddings,
    similarity_threshold=0.8,
)
print(graph.adjacency)
```

### گراف معنایی با FarsNet

FarsNet، WordNet فارسی است و می‌تواند منبع قوی‌تری برای ساخت رابطه‌های
معنایی بین واژه‌های فارسی باشد. به دلیل مسائل دسترسی و لایسنس، دیتابیس
FarsNet داخل این مخزن کپی نشده است؛ اما پروژه loader آماده دارد تا اگر خروجی
FarsNet را به شکل JSON/CSV/TSV داشته باشید، مستقیم از آن گراف معنایی بسازد.

نمونه JSON قابل استفاده:

```json
{
  "synsets": [
    {"id": "s1", "lemmas": ["ماشین", "خودرو", "اتومبیل"]},
    {"id": "s2", "lemmas": ["پزشک", "دکتر"]}
  ]
}
```

استفاده در پروژه:

```python
from rakhshai_graph_nlp.graphs.semantic import (
    build_semantic_graph_from_farsnet,
)

words = ["ماشین", "خودرو", "پزشک", "دکتر"]

graph = build_semantic_graph_from_farsnet(
    words,
    farsnet_path="data/farsnet.json",
)
print(graph.adjacency)
```

اگر خروجی FarsNet شما CSV/TSV باشد، دو حالت رایج پشتیبانی می‌شود:

```csv
source,target
ماشین,خودرو
پزشک,دکتر
```

یا:

```csv
synset_id,word
s1,ماشین
s1,خودرو
s2,پزشک
s2,دکتر
```

## ساختار پروژه

```text
rakhshai_graph_nlp/
├── features/        # توکنایز، پیش‌پردازش و تبدیل به PyG
├── graphs/          # توابع ساخت گراف
├── models/          # مدل‌های GNN
├── lm/              # PersianTokenizer، LMDataset، GraphCausalLM، Graph Memory، LMTrainer و generate
├── llm/             # workflowهای بومی سطح بالا که روی موتور Graph-LM ساخته شده‌اند
├── article_llm/     # مسیر سازگاری برای workflow مقاله‌نویسی بومی
├── tasks/           # وظایف کاربردی
├── explain/         # ابزارهای تبیین اولیه
├── metrics.py       # معیارهای ارزیابی
├── cli.py           # رابط خط فرمان
└── utils/           # توابع کمکی

benchmarks/          # دیتاست‌های کوچک قابل تکرار
docs/                # مستندات MkDocs
tests/               # تست‌های واحد و end-to-end
```

## اعتبارسنجی و کیفیت

`RGN` همراه با تست‌های واحد، یکپارچه، CLI، checkpoint، گراف و end-to-end
منتشر می‌شود. دیتاست‌های کوچک در `benchmarks/` و `data/` نیز برای بررسی
پیاده‌سازی و اجرای تکرارپذیر محلی در دسترس‌اند.

```bash
python -m pytest
```

جزئیات تکمیلی معماری، روش `benchmark` و ارزیابی فنی در مستندات تخصصی آمده‌اند:

- [Graph-LM V2](docs/graph_lm_v2.md)
- [Graph Reasoning Core](docs/graph_reasoning_core.md)
- [Low-Data Training Engine](docs/low_data_training_engine.md)
- [گزارش ارزیابی MCP](docs/mcp_single_poem_evaluation.md)

## نکات استقرار و استفادهٔ مسئولانه

- کیفیت تولید به پیکرهٔ آموزش، توکن‌ساز، چک‌پوینت، تنظیمات تولید و رمزگذار و
  روش ترکیب گراف انتخاب‌شده وابسته است.
- `build_text_graph` از ماتریس dense استفاده می‌کند و برای مجموعه‌های بسیار
  بزرگ ممکن است به مسیر مقیاس‌پذیرتری برای ساخت گراف نیاز داشته باشد.
- مدل هامان یک چک‌پوینت پایه برای تولید مقاله است، نه چت‌بات یا مرجع
  واقعیت؛ خروجی را پیش از انتشار بازبینی کنید.
- طبقه‌بندهای حساس، از جمله تشخیص نفرت‌پراکنی، باید پیش از استفاده با دادهٔ
  نماینده، تحلیل خطا و کنترل سوگیری آموزش و ارزیابی شوند.
- کلیدها و اطلاعات محرمانه را بیرون از مخزن نگه دارید و سیاست‌های هر سرویس
  بیرونی متصل از طریق `MCP` یا اتصال‌های اختیاری را بررسی کنید.

## مستندات

| موضوع | راهنما |
| --- | --- |
| فهرست مستندات | [`docs/index.md`](docs/index.md) |
| API پایدار پایتون | [`docs/api.md`](docs/api.md) |
| راهنمای گام‌به‌گام API | [`docs/api_usage.md`](docs/api_usage.md) |
| گردش‌کارهای بومی `LLM` | [`docs/llm.md`](docs/llm.md) |
| گردش‌کار مقاله‌نویسی فارسی | [`docs/article_llm.md`](docs/article_llm.md) |
| معماری `Graph-LM` | [`docs/graph_lm_v2.md`](docs/graph_lm_v2.md) |
| هستهٔ استدلال گرافی | [`docs/graph_reasoning_core.md`](docs/graph_reasoning_core.md) |
| موتور آموزش دادهٔ کم | [`docs/low_data_training_engine.md`](docs/low_data_training_engine.md) |
| توکن‌ساز فارسی | [`docs/persian_tokenizer.md`](docs/persian_tokenizer.md) |
| گراف چندرابطه‌ای | [`docs/multi_relation_persian_graph.md`](docs/multi_relation_persian_graph.md) |
| اتصال MCP | [`docs/mcp.md`](docs/mcp.md) |

## دربارهٔ توسعه‌دهنده

ما `RGN` را در تیم [RakhshAI](https://rakhshai.com/) شرکت دانش‌بنیان
[آریا هامان مهر پارسه](https://ariahaman.ir/) ایجاد و نگهداری می‌کنیم.

### تأییدیهٔ دانش‌بنیان

محصول ما، `Rakhshai Graph-based NLP (RGN)`، تأییدیهٔ دانش‌بنیان را دریافت
کرده است. برای
استعلام، شناسهٔ ملی شرکت را در
[سامانهٔ رسمی شرکت‌های دانش‌بنیان](https://pub.daneshbonyan.ir/dashboard)
جست‌وجو کنید.

| مشخصه | مقدار |
| --- | --- |
| شرکت | آریا هامان مهر پارسه |
| شناسهٔ ملی | `14009192677` |
| محصول | `Rakhshai Graph-based NLP (RGN)` |
| حوزه | پردازش زبان طبیعی فارسی، مدل‌سازی گرافی متن و زیرساخت هوش مصنوعی |

## مجوزها

- کد منبع `RGN`: [MIT](LICENSE)
- مدل هامان:
  [مجوز `Haman Model License 1.0`](https://huggingface.co/aria-haman/haman-fa-article-graph-llm-125m/blob/main/LICENSE)
- دیتاست آموزش مدل هامان: مجوز `CC BY-SA 4.0`؛ جزئیات در
  [کارت دیتاست](https://huggingface.co/datasets/aria-haman/haman-fa-wikipedia-articles-186k)

برای سؤال فنی و گزارش خطا از
[GitHub Issues](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/issues)
استفاده کنید.
