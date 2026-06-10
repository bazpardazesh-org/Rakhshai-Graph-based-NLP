#  نخستین چارچوب و فریم‌ورک یکپارچه Graph-based NLP برای زبان فارسی

[![CI](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/bazpardazesh-org/Rakhshai-Graph-based-NLP/actions/workflows/ci.yml)

**Rakhshai Graph-based NLP**؛ از متن خام فارسی تا شبکه‌های عصبی گرافی.

رخشای نخستین فریم‌ورک یکپارچه NLP گراف‌محور برای زبان فارسی است؛ چارچوبی کاربردی
که متن خام فارسی را به گراف‌های قابل تحلیل و قابل یادگیری تبدیل می‌کند. این پروژه
رابطهٔ واژه‌ها، اسناد و ساختارهای زبانی را در قالب گراف مدل‌سازی می‌کند و امکان
آموزش و استفاده از مدل‌های شبکه عصبی گرافی مانند GCN، GraphSAGE و GAT را برای
وظایفی مانند طبقه‌بندی متن، خلاصه‌سازی، توصیه‌گر محتوا و تحلیل‌های متنی فراهم می‌سازد.

رخشای برای کوتاه‌کردن مسیر ساخت pipelineهای Graph-based NLP در زبان فارسی طراحی شده است:
از پیش‌پردازش متن فارسی و ساخت گراف‌های متنی، نحوی و معنایی تا آموزش، ارزیابی،
پیش‌بینی روی متن جدید و ذخیره/بارگذاری مدل. هدف رخشای این است که پلی عملی میان
زبان فارسی، مدل‌سازی گرافی و یادگیری عمیق گرافی ایجاد کند.

از نسخهٔ جدید، رخشای فقط یک ابزار طبقه‌بندی گرافی نیست؛ یک مسیر واقعی
**Persian Graph-LM** هم دارد. در این مسیر، متن فارسی به توکن عددی تبدیل
می‌شود، از همان corpus گراف هم‌رخدادی واژگان ساخته می‌شود، GNN روی گراف
embedding گرافی تولید می‌کند، و سپس embedding توکن و embedding گراف با
**Gated Graph-Token Fusion** درون یک Transformer causal language model ترکیب
می‌شوند. خروجی این مدل `batch × sequence × vocab_size` است و برای پیش‌بینی
توکن بعدی و تولید متن فارسی طراحی شده است.

در آزمایش اولیه روی یک corpus فارسی توسعه‌یافته، مسیر `Graph-LM / GCN + gated`
نسبت به baseline بدون گراف perplexity پایین‌تری گرفت. این نتیجه نشان می‌دهد
که استفاده از رابطه‌های واژگانی گرافی می‌تواند در این setup به پیش‌بینی بهتر
توکن بعدی کمک کند؛ با این حال این مسیر هنوز experimental است و کیفیت نهایی
به اندازه داده، تنظیمات آموزش و ارزیابی‌های بزرگ‌تر وابسته است.


این کتابخانه متن‌باز توسط تیم توسعه [RakhshAI](https://rakhshai.com/) وابسته به
شرکت آریا هامان مهر پارسه ایجاد و توسعه داده شده و تحت مجوز MIT منتشر می‌شود.

## ویژگی‌های برجسته به زبان ساده

- **نخستین فریم‌ورک یکپارچه NLP گراف‌محور برای زبان فارسی:**  
  رخشای یک مسیر کامل از متن خام فارسی تا ساخت گراف، آموزش مدل، ارزیابی،
  پیش‌بینی و ذخیره/بارگذاری pipeline فراهم می‌کند.
- **پیاده‌سازی اختصاصی GCN، GraphSAGE و GAT برای NLP گراف‌محور فارسی:**  
  رخشای سه مدل مهم شبکه عصبی گرافی، یعنی GCN، GraphSAGE و GAT را در قالب
  یک فریم‌ورک اجرایی و یکپارچه برای یادگیری روی گراف‌های ساخته‌شده از متن فارسی
  ارائه می‌کند؛ با امکان استفاده در آموزش، ارزیابی، پیش‌بینی و ذخیره/بارگذاری pipeline.
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
- **Graph-LM فارسی با معماری اختصاصی Rakhshai:** مسیر `lm-train` یک tokenizer
  فارسی، graph builder، GNN encoder، gated graph-token fusion، Transformer
  causal LM، trainer مخصوص LM، perplexity، checkpoint کامل شامل artifact گراف
  sparse و تولید متن را کنار هم قرار می‌دهد.
- **Graph Reasoning Core برای گراف چندرابطه‌ای:** encoder گراف می‌تواند
  relation idهای فاز ۳ را با حالت‌های `bias`، `embedding` یا `rgcn` مصرف کند،
  از `RGCN` برای message passing رابطه‌محور استفاده کند، و به صورت اختیاری
  node importance و subgraph pooling داشته باشد.
- **Adaptive Graph-Text Fusion:** مدل می‌تواند ترکیب متن و گراف را در سطح
  `token`، `sentence` و `subgraph` کنترل کند. شدت استفاده از گراف با
  scale/dropout تنظیم می‌شود و آمار gateها در `metrics.json` ذخیره می‌شود تا
  معلوم شود مدل واقعاً کجا از گراف کمک گرفته است.
- **Low-Data Training Engine فاز ۷:** مسیر Graph-LM به صورت پیش‌فرض با
  augmentation متنی، dropout گره/یال، subgraph sampling، contrastive learning،
  curriculum learning، early stopping و گزارش overfitting اجرا می‌شود تا در
  corpusهای کوچک کمتر حفظ کند و بهتر generalize کند.
- **Graph Memory فاز ۸ برای زمان تولید:** checkpointهای گرافی اکنون حافظهٔ
  گرافی جداگانه ذخیره می‌کنند. دستور `generate` به صورت پیش‌فرض از prompt
  nodeها و subgraphهای مرتبط را retrieve می‌کند و همان زیرگراف محدود را به
  fusion می‌دهد تا مدل به جای کل گراف، حافظهٔ مرتبط با prompt را مصرف کند.
- **Baseline بدون گراف برای مقایسه منصفانه:** با `--graph-encoder none` می‌توانید
  همان Transformer causal LM را بدون GNN و fusion آموزش دهید و اثر واقعی گراف را
  با validation loss و perplexity بسنجید.
- **تولید متن قابل کنترل:** دستور `generate` از گزینه‌هایی مثل `--temperature`،
  `--top-k`، `--min-new-tokens` و `--repetition-penalty` پشتیبانی می‌کند.

## امضای فنی Graph-LM رخشای

معماری جدید Graph-LM در رخشای این مسیر را پیاده‌سازی می‌کند:

```text
Persian Text
→ PersianTokenizer
→ LM Dataset
→ Multi-Relation Persian Graph
→ Rakhshai Graph Encoder (GCN / GraphSAGE / GAT / RGCN)
→ Adaptive Graph-Text Fusion
→ Low-Data Training Engine
→ Prompt-aware Graph Memory
→ Transformer Causal LM
→ Text Generation
```

امضای فنی پروژه در این بخش:

```text
Rakhshai Graph Encoder
+
Adaptive Graph-Text Fusion
+
Low-Data Training Engine
+
Persian Causal LM
```

به زبان ساده، رخشای می‌تواند به جای یک مدل زبانی صرفاً دنباله‌ای، از رابطهٔ
گرافی واژه‌ها هم استفاده کند. مدل هنگام ساخت embedding نهایی هر توکن یاد
می‌گیرد چقدر به embedding متنی و چقدر به embedding گرافی اعتماد کند؛ یعنی
ترکیب به صورت ثابت و دستی نیست، بلکه با gate قابل یادگیری انجام می‌شود.
این gate می‌تواند در چند سطح فعال شود: سطح توکن برای هر جایگاه دنباله، سطح
جمله برای کنترل شدت کلی گراف در context، و سطح subgraph برای تزریق خلاصه‌ای از
nodeهای غیرتوکنی مانند سند یا topic.

## نتیجهٔ اولیه Graph-LM

برای اعتبارسنجی مسیر جدید، یک corpus فارسی توسعه‌یافته ساخته شد و دو مدل با
شرایط مشابه مقایسه شدند: مدل گرافی `Graph-LM / GCN + gated` و baseline بدون
گراف. نتیجهٔ آزمایش کوچک اولیه:

| مدل | best validation loss | best perplexity |
| --- | ---: | ---: |
| `Graph-LM / GCN + gated` | 7.2040 | 1344.7671 |
| `Baseline / no graph` | 7.3994 | 1634.9926 |

در این اجرا، Graph-LM حدود ۱۸٪ perplexity پایین‌تری نسبت به baseline بدون گراف
داشت. 

نمونهٔ خروجی تولید متن با corpus کوچک‌تر از مدل‌های بزرگ:

```text
prompt: امروز در تهران
output: امروز در تهران باران آرامی بارید و خیابان‌ها خلوت‌تر از روزهای گذشته بودند ...
```

برای نتیجهٔ علمی‌تر، پیشنهاد می‌شود همین آزمایش با چند seed، چند graph encoder
مثل `GCN`، `GAT`، `GraphSAGE` و `RGCN`، چند حالت relation-aware مثل
`bias` و `embedding`، و چند روش fusion مثل `gated` و `additive` تکرار شود.

## نمونه دیتاست آماده برای تست Graph-LM

یک نمونه آماده از Persian Wikipedia برای تست سریع Graph-LM در مسیر زیر قرار دارد:

```text
data/wiki_fa_50k.txt
```

اگر نمی‌خواهید دیتاست را جداگانه دانلود کنید، می‌توانید مستقیم با همین فایل
آموزش و تولید متن را تست بگیرید. برای ساخت دوباره همین فایل، یا بزرگ‌تر کردن
نمونه، از اسکریپت زیر استفاده کنید:

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

در این دستور، Graph Memory به صورت پیش‌فرض فعال است. اگر checkpoint شامل
`graph_memory.pt` باشد، همان حافظه بارگذاری می‌شود؛ اگر نباشد و `corpus.txt`
در checkpoint وجود داشته باشد، حافظه از corpus و `graph_config.json` بازسازی
می‌شود. برای خاموش‌کردن حافظه و برگشت به generate قبلی:

```bash
rgnn-cli generate \
  --model runs/wiki-graph-lm \
  --prompt "امروز در تهران" \
  --max-new-tokens 100 \
  --graph-memory off
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
| `PersianTokenizer` | tokenizer عددی مخصوص LM با پشتیبانی از نیم‌فاصله، تمیزسازی فارسی و نرمال‌سازی «ی/ي» و «ک/ك» |
| `LMDataset` | آماده‌سازی `input_ids` و `target_ids` برای پیش‌بینی توکن بعدی |
| `build_graph_lm_graph` | ساخت گراف هم‌رخدادی واژگان از corpus برای Graph-LM |
| `GraphCausalLM` | مدل زبانی فارسی با GNN encoder، gated graph-token fusion و Transformer causal LM |
| `RakhshaiGraphEncoder` | هستهٔ گرافی Graph-LM با پشتیبانی از `gcn`، `graphsage`، `gat`، `rgcn` و relation-aware encoding |
| `--graph-encoder none` | baseline بدون گراف برای مقایسه با Graph-LM و سنجش اثر واقعی GNN/fusion |
| `LMTrainer` | trainer مخصوص LM با validation loss، perplexity و checkpoint کامل |
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
python -m pip install -e ".[ml]"    # ابزارهای scikit-learn برای گراف سند و توصیه‌گر
python -m pip install -e ".[nlp]"   # پشتیبانی Stanza برای گراف وابستگی نحوی
python -m pip install -e ".[fa]"    # پشتیبانی Faiss
python -m pip install -e ".[docs]"  # ابزار ساخت مستندات
python -m pip install -e ".[dev]"   # ابزار تست، lint، build و انتشار
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

ابزار `rgnn-cli` دو مسیر اصلی دارد: مسیر قدیمی و پایدار طبقه‌بندی متن فارسی
مبتنی بر گراف، و مسیر جدید Graph-LM برای آموزش مدل زبانی فارسی گراف‌محور.
اگر subcommand ندهید، همان مسیر طبقه‌بندی اجرا می‌شود. برای LM از
`lm-train` و `generate` استفاده کنید.

آموزش Graph-LM:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --fusion gated \
  --output-dir runs/graph-lm
```

در حالت پیش‌فرض، مسیر Graph-LM گراف چندرابطه‌ای فاز ۳ را می‌سازد:

```text
cooccurrence + pmi + stem + subword + word_document + topic_document
```

برای بازتولید baseline سادهٔ قدیمی فاز ۱، relation را صریح محدود کنید:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder gcn \
  --graph-relations cooccurrence \
  --output-dir runs/v1-simple-graph
```

برای فعال‌کردن Graph Reasoning Core فاز ۴، می‌توانید relationها را به شکل
غنی‌تر به encoder بدهید. نمونهٔ سبک با GraphSAGE رابطه‌محور:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-encoder graphsage \
  --graph-relation-mode embedding \
  --graph-pooling attention \
  --graph-node-importance \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --output-dir runs/phase4-graphsage-reasoning
```

نمونهٔ مستقیم با R-GCN:

```bash
rgnn-cli lm-train \
  --corpus data/expanded_persian_lm.txt \
  --graph-relation-mode rgcn \
  --graph-relations cooccurrence pmi stem word_document topic_document \
  --output-dir runs/phase4-rgcn
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

از فاز ۷، **Low-Data Training Engine** به صورت پیش‌فرض فعال است. این بخش
برای corpusهای کوچک طراحی شده و با augmentation متنی، dropout گرافی،
subgraph sampling، contrastive consistency، curriculum learning و early
stopping تلاش می‌کند مدل کمتر متن را حفظ کند و validation بهتری بگیرد.

نمونه اجرای صریح فاز ۷ با تنظیمات پیشنهادی پیش‌فرض:

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

خروجی checkpoint مدل زبانی کامل است و فقط به `model.pt` محدود نمی‌شود:

```text
runs/graph-lm/
├── model.pt
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
| `--graph-encoder` | انتخاب `gcn`، `gat`، `graphsage`، `rgcn` یا `none` برای baseline بدون گراف |
| `--graph-relations` | انتخاب رابطه‌های گراف؛ اگر ندهید preset چندرابطه‌ای فاز ۳ فعال است |
| `--relation-weights` | وزن‌دهی relationها، مثل `pmi=0.7,stem=0.5` |
| `--graph-relation-mode` | نحوهٔ مصرف `edge_type` در encoder؛ یکی از `bias`، `embedding` یا `rgcn` |
| `--graph-pooling` | pooling اختیاری روی subgraphها؛ یکی از `none`، `mean` یا `attention` |
| `--graph-node-importance` | فعال‌کردن scorer داخلی برای تشخیص nodeهای مهم‌تر |
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
| `--augmentation-ratio` | نسبت نمونه‌های augmented اضافه‌شده به train split؛ پیش‌فرض فاز ۷ فعال است |
| `--token-dropout` و `--punctuation-dropout` | شدت augmentation متنی برای corpus کم‌داده |
| `--edge-dropout` و `--node-dropout` | regularization گرافی هنگام train برای جلوگیری از حفظ ساختار ثابت |
| `--subgraph-sampling-ratio` | سهم یال‌های نگه‌داشته‌شده در هر view گرافی train |
| `--contrastive-weight` | وزن loss سازگاری contrastive بین viewهای گرافی |
| `--no-text-augmentation` و `--no-curriculum` | خاموش‌کردن augmentation متنی یا curriculum برای ablation |
| `--early-stopping-patience` و `--early-stopping-min-delta` | کنترل early stopping بر اساس validation loss |
| `--max-grad-norm` | حد clipping گرادیان در trainer |
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
├── tasks/           # وظایف کاربردی
├── explain/         # ابزارهای تبیین اولیه
├── metrics.py       # معیارهای ارزیابی
├── cli.py           # رابط خط فرمان
└── utils/           # توابع کمکی

benchmarks/          # دیتاست‌های کوچک قابل تکرار
docs/                # مستندات MkDocs
tests/               # تست‌های واحد و end-to-end
```

## benchmark اولیه Graph-LM

برای بررسی مسیر Graph-LM، یک corpus فارسی توسعه‌یافته در `data/expanded_persian_lm.txt`
استفاده شد. این corpus شامل جمله‌هایی درباره شهر، سیاست، آموزش، اقتصاد، ورزش،
هنر و خود معماری Graph-LM است تا رابطه‌های واژگانی مثل `مجلس/قانون`،
`تهران/باران`، `مدرسه/دانش‌آموز` و `مدل/گراف/embedding` در گراف هم‌رخدادی
قابل مشاهده باشند.

مقایسهٔ اولیه:

| مدل | best validation loss | best perplexity | مسیر خروجی |
| --- | ---: | ---: | --- |
| `Graph-LM / GCN + gated` | 7.2040 | 1344.7671 | `runs/compare-graph-lm` |
| `Baseline / no graph` | 7.3994 | 1634.9926 | `runs/compare-baseline-lm` |

نمونه تولید متن با مدل demo:

```text
prompt: امروز در تهران
output: امروز در تهران باران آرامی بارید و خیابان‌ها خلوت‌تر از روزهای گذشته بودند ...
```

این benchmark کوچک برای اثبات مسیر و مقایسهٔ اولیه است. 

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
- مسیر Graph-LM فعلاً experimental است. benchmark اولیه نشان‌دهندهٔ درست‌بودن
  pipeline و اثر مثبت اولیهٔ گراف است، نه کیفیت نهایی در سطح LLMهای بزرگ.
- کیفیت تولید متن در Graph-LM به اندازه و تنوع corpus، تعداد epoch، tokenizer،
  تنظیمات sampling و انتخاب graph encoder/fusion وابسته است.
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
- **Causal Language Modeling:** آموزش مدل برای پیش‌بینی توکن بعدی و تولید متن.
- **Transformer Decoder:** هستهٔ مدل زبانی دنباله‌ای که با graph-token fusion
  از embedding گرافی هم استفاده می‌کند.
- **Stanza:** ابزار تحلیل زبانی برای tokenization، lemmatization و dependency
  parsing.
