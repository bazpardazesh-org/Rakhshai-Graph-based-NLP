# Rakhshai Graph-based NLP for Persian

[![CI](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml)

**Rakhshai Graph-based NLP** یک کتابخانه کاربردی برای پردازش زبان فارسی با
نگاه گرافی است. این پروژه متن فارسی را به گراف تبدیل می‌کند، رابطهٔ واژه‌ها
و سندها را مدل می‌کند و سپس با مدل‌هایی مثل GCN، GraphSAGE و GAT روی آن
طبقه‌بندی، خلاصه‌سازی، توصیه‌گر محتوا و تحلیل‌های متنی انجام می‌دهد.

اگر دنبال یک ابزار فارسی‌محور برای ساخت pipelineهای NLP مبتنی بر گراف هستید،
این پروژه تلاش می‌کند مسیر را کوتاه کند: از متن خام فارسی تا ساخت گراف،
آموزش مدل، ارزیابی، پیش‌بینی روی متن جدید و ذخیره/بارگذاری مدل.

این نسخه غیرتجاری توسط تیم توسعه [RakhshAI](https://rakhshai.com/) وابسته به
شرکت آریا هامان مهر پارسه ایجاد و توسعه داده شده است.

## ویژگی‌های برجسته به زبان ساده

- **فارسی‌محور از ابتدا:** مثال‌ها، توکن‌سازی، پیش‌پردازش و کاربردها با متن
  فارسی طراحی شده‌اند.
- **طبقه‌بندی متن آماده استفاده:** با `TextGraphClassifier` می‌توانید متن و
  برچسب بدهید، مدل آموزش دهید، ارزیابی کنید و روی متن جدید خروجی بگیرید.
- **ذخیره کامل مدل و pipeline:** فقط وزن مدل ذخیره نمی‌شود؛ واژگان، نگاشت
  برچسب‌ها و تنظیمات graph هم همراه مدل ذخیره می‌شوند.
- **پشتیبانی از GPU برای مدل‌های گرافی:** اگر CUDA و کارت NVIDIA داشته باشید،
  مسیرهای PyTorch Geometric می‌توانند روی GPU اجرا شوند.
- **چند نوع گراف برای متن فارسی:** هم‌رخدادی واژه‌ها، گراف واژه-سند،
  شباهت اسناد، وابستگی نحوی و گراف معنایی با پشتیبانی از FarsNet.
- **چند وظیفه آماده:** طبقه‌بندی، خلاصه‌سازی، توصیه‌گر محتوا، تشخیص
  نفرت‌پراکنی و تحلیل شبکه.
- **رابط خط فرمان:** با `rgnn-cli` می‌توانید بدون نوشتن کد زیاد، روی فایل
  CSV/TSV/JSONL آموزش و ارزیابی انجام دهید.

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

| مؤلفه | توضیح |
| --- | --- |
| `TextGraphClassifier` | کلاس اصلی برای آموزش، ارزیابی، پیش‌بینی، ذخیره و بارگذاری مدل طبقه‌بندی متن |
| `build_text_graph` | ساخت گراف واژه-سند با PMI و TF-IDF برای TextGCN |
| `build_cooccurrence_graph` | ساخت گراف هم‌رخدادی واژه‌ها در پنجرهٔ لغتی |
| `build_document_graph` | ساخت گراف شباهت سندها با TF-IDF یا embedding |
| `build_dependency_graph` | ساخت گراف وابستگی نحوی با Stanza |
| `build_semantic_graph` | ساخت گراف معنایی از روابط واژگانی و شباهت embedding |
| `build_semantic_graph_from_farsnet` | ساخت گراف معنایی از خروجی JSON/CSV/TSV مربوط به FarsNet |
| `load_farsnet_relations` | خواندن روابط FarsNet از فایل و تبدیل آن به روابط قابل استفاده در گراف |
| `GCNClassifier` | مدل GCN برای طبقه‌بندی گره‌ها |
| `GraphSAGEClassifier` | مدل GraphSAGE برای یادگیری از ساختار همسایگی |
| `GATClassifier` | مدل GAT با مکانیزم attention روی گراف |
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
python -m pip install -e ".[sparse]"  # ابزارهای گراف تنک
python -m pip install -e ".[fa]"      # پشتیبانی Faiss
python -m pip install -e ".[docs]"    # ابزار ساخت مستندات
```

### اجرای تست‌ها

```bash
python -m pip install pytest
python -m pytest
```

## شروع سریع

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

ابزار `rgnn-cli` برای اجرای سریع مسیر طبقه‌بندی متن فارسی مبتنی بر گراف است.
اگر `--dataset` ندهید، یک آزمایش داخلی کوچک اجرا می‌شود تا نصب و مدل پایه
را smoke test کنید. اگر دیتاست بدهید، CLI متن‌ها را می‌خواند، گراف واژه-سند
می‌سازد، یکی از مدل‌های `gcn`، `graphsage` یا `gat` را آموزش می‌دهد و
گزارش train/validation/test می‌نویسد.

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
├── tasks/           # وظایف کاربردی
├── explain/         # ابزارهای تبیین اولیه
├── metrics.py       # معیارهای ارزیابی
├── cli.py           # رابط خط فرمان
└── utils/           # توابع کمکی

benchmarks/          # دیتاست‌های کوچک قابل تکرار
docs/                # مستندات MkDocs
tests/               # تست‌های واحد و end-to-end
```

## benchmarkهای قابل تکرار

یک benchmark کوچک فارسی در `benchmarks/persian_text_classification.csv` وجود
دارد تا مسیر کامل ساخت گراف، آموزش، ارزیابی و ذخیره گزارش سریع بررسی شود.
این دیتاست ۲۴ متن خبری کوتاه در سه کلاس `politics`، `sports` و `art` دارد؛
برای smoke test و مقایسه تنظیمات مناسب است، نه برای ادعای کیفیت نهایی مدل.

نمونه اجراهای تکرارپذیر روی CPU با `seed=0`:

| مدل | validation accuracy | test accuracy | test macro-F1 | مسیر گزارش |
| --- | ---: | ---: | ---: | --- |
| `gcn` | 1.00 | 0.75 | 0.60 | `runs/benchmarks/persian-classification-gcn/metrics.json` |
| `graphsage` | 1.00 | 1.00 | 1.00 | `runs/benchmarks/persian-classification-graphsage/metrics.json` |
| `gat` | 1.00 | 1.00 | 1.00 | `runs/benchmarks/persian-classification-gat/metrics.json` |

نمونه خروجی `metrics.json`:

```json
{
  "dataset": "benchmarks/persian_text_classification.csv",
  "model": "gcn",
  "device": "cpu",
  "num_documents": 24,
  "num_nodes": 175,
  "num_classes": 3,
  "splits": {
    "train": {"count": 16, "accuracy": 1.0, "macro_f1": 1.0},
    "validation": {"count": 4, "accuracy": 1.0, "macro_f1": 1.0},
    "test": {"count": 4, "accuracy": 0.75, "macro_f1": 0.6}
  }
}
```

برای اجرای همان benchmark توسط خودتان:

```bash
python -m rakhshai_graph_nlp.cli \
  --dataset benchmarks/persian_text_classification.csv \
  --model gcn \
  --epochs 50 \
  --hidden-dim 8 \
  --learning-rate 0.01 \
  --dropout 0.2 \
  --seed 0 \
  --device cpu \
  --output-dir runs/benchmarks/persian-classification-gcn
```

برای مقایسه سه مدل اصلی:

```bash
for model in gcn graphsage gat; do
  python -m rakhshai_graph_nlp.cli \
    --dataset benchmarks/persian_text_classification.csv \
    --model "$model" \
    --epochs 50 \
    --hidden-dim 8 \
    --learning-rate 0.01 \
    --dropout 0.2 \
    --seed 0 \
    --device cpu \
    --output-dir "runs/benchmarks/persian-classification-$model"
done
```

پس از اجرا، گزارش هر مدل در `metrics.json` ذخیره می‌شود. برای دیدن خلاصه:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("runs/benchmarks").glob("*/metrics.json")):
    report = json.loads(path.read_text(encoding="utf-8"))
    test = report["splits"]["test"]
    print(
        f"{report['model']:10s} "
        f"test_acc={test['accuracy']:.3f} "
        f"test_macro_f1={test['macro_f1']:.3f} "
        f"report={path}"
    )
PY
```

برای سنجش جدی‌تر، همین pipeline را روی benchmarkهای شناخته‌شده فارسی هم اجرا
کنید. کافی است دیتاست را به CSV/TSV/JSONL با ستون‌های `text` و `label`
تبدیل کنید و همان دستور CLI را بدهید.

| benchmark | کاربرد رایج | نکته آماده‌سازی |
| --- | --- | --- |
| [Hamshahri Corpus](https://en.wikipedia.org/wiki/Hamshahri_Corpus) | طبقه‌بندی خبر و بازیابی اطلاعات فارسی | متن خبر را در `text` و دسته خبر را در `label` بگذارید. |
| [SnappFood Persian Sentiment](https://www.kaggle.com/datasets/soheiltehranipour/snappfood-persian-sentiment-analysis) | تحلیل احساسات نظر کاربران | متن نظر را `text` و برچسب مثبت/منفی را `label` کنید. |
| [SentiPers](https://www.researchgate.net/publication/322694676_SentiPers_A_Sentiment_Analysis_Corpus_for_Persian) | تحلیل احساسات فارسی | اگر چند سطح polarity دارید، آن‌ها را به برچسب‌های متنی ثابت تبدیل کنید. |
| [Pars-ABSA](https://arxiv.org/abs/1908.01815) | احساسات جنبه‌محور فارسی | برای این CLI، هر نمونه را به یک برچسب کلی تبدیل کنید یا هر aspect را یک ردیف جدا بگیرید. |

مثال اجرای دیتاست خودتان:

```bash
python -m rakhshai_graph_nlp.cli \
  --dataset data/my_persian_dataset.csv \
  --text-column text \
  --label-column label \
  --model gat \
  --epochs 80 \
  --hidden-dim 16 \
  --learning-rate 0.005 \
  --dropout 0.3 \
  --seed 42 \
  --device cuda \
  --output-dir runs/my-persian-benchmark-gat
```

اگر CUDA در سیستم در دسترس نباشد، `--device cpu` بگذارید. برای مقایسه منصفانه
بین مدل‌ها، split، seed، تعداد epoch و پیش‌پردازش را ثابت نگه دارید و علاوه
بر accuracy، مقدار macro-F1 را هم گزارش کنید؛ مخصوصاً وقتی کلاس‌ها نامتوازن
هستند.

## محدودیت‌های فعلی

- `build_text_graph` از ماتریس dense استفاده می‌کند و برای مجموعه‌های خیلی
  بزرگ می‌تواند از نظر حافظه محدود شود.
- کیفیت خروجی مدل‌ها به کیفیت داده، توکن‌سازی و تنظیمات آموزش وابسته است.
- `HateSpeechDetector` برای کاربرد حساس باید با دادهٔ واقعی، بررسی خطا و
  کنترل bias آموزش داده شود.
- برای گراف معنایی قوی‌تر، از FarsNet، روابط واژگانی معتبر یا embedding
  فارسی باکیفیت استفاده کنید.

## منابع الهام

- **TextGCN:** گراف واژه-سند با PMI و TF-IDF برای طبقه‌بندی متن.
- **GCN / GraphSAGE / GAT:** مدل‌های شبکهٔ عصبی گرافی برای انتشار و تجمیع
  اطلاعات در گراف.
- **TextRank:** رتبه‌بندی جمله‌ها یا واژه‌ها با PageRank روی گراف شباهت.
- **Stanza:** ابزار تحلیل زبانی برای tokenization، lemmatization و dependency
  parsing.
