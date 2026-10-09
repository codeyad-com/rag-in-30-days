<div dir="rtl" align="right">

# 🤖 ساخت دستیار هوشمند اختصاصی در ۳۰ روز

**پروژه‌ی عملی دوره‌ی «ساخت دستیار هوشمند اختصاصی در ۳۰ روز» — آکادمی کدیاد**

[![Academy](https://img.shields.io/badge/Academy-Codeyad-6C5CE7?style=for-the-badge)](https://codeyad.com)
[![Course](https://img.shields.io/badge/Course-Build%20Custom%20AI%20Assistant%20in%2030%20Days-00B894?style=for-the-badge)](https://codeyad.com/expert/build-custom-ai-assistant-in-30-days)
[![Students](https://img.shields.io/badge/Students-100%2B-blue?style=for-the-badge)](#)
[![Level](https://img.shields.io/badge/Level-Advanced-red?style=for-the-badge)](#)
[![Hours](https://img.shields.io/badge/Duration-20%2B%20Hours-orange?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=for-the-badge)](#)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C?logo=qdrant&logoColor=white)
![RAG](https://img.shields.io/badge/Architecture-RAG-2D3436)

---

## 📖 درباره‌ی این پروژه

در دوره‌ی **«ساخت دستیار هوشمند اختصاصی در ۳۰ روز»** فقط با مفاهیم RAG آشنا نمی‌شوید؛ بلکه قدم‌به‌قدم یک دستیار هوشمند واقعی برای پاسخ‌گویی به اسناد سازمانی می‌سازید. از پردازش فایل‌های PDF و جست‌وجوی هوشمند تا طراحی پنل چت و مدیریت، همه‌چیز را به‌صورت پروژه‌محور پیاده‌سازی می‌کنید.

کسب‌وکارها اطلاعات زیادی در فایل‌ها، قوانین، مستندات و سوالات متداول خود دارند؛ اما پیدا کردن پاسخ درست میان این اطلاعات همیشه ساده نیست. در این پروژه یک دستیار هوشمند اختصاصی ساخته می‌شود که:

1. اسناد سازمانی (PDF) را می‌خواند و پردازش می‌کند.
2. اطلاعات مرتبط با سوال کاربر را با جست‌وجوی برداری پیدا می‌کند.
3. پاسخ دقیق را **همراه با منبع و شماره صفحه** در اختیار کاربر قرار می‌دهد.

این سورس یک تمرین آموزشی نیست؛ یک **محصول قابل ارائه به کسب‌وکار** است که می‌توانید آن را شخصی‌سازی کنید.

> 🎯 **مخاطب دوره:** افرادی که با مقدمات پایتون آشنا هستند و می‌خواهند وارد دنیای ساخت محصولات واقعی هوش مصنوعی شوند.

---

## ✨ ویژگی‌های پروژه

- 📄 **پردازش اسناد PDF** با `PyMuPDF` (استخراج متن صفحه‌به‌صفحه)
- 🧹 **پاک‌سازی و پیش‌پردازش متن فارسی** (حذف شماره‌گذاری‌ها، ادغام خطوط شکسته، نرمال‌سازی فاصله‌ها)
- ✂️ **قطعه‌بندی هوشمند (Chunking)** با توکن‌سایزر جملات فارسی `hazm` و پشتیبانی از **همپوشانی (Overlap)**
- 🔢 **دو حالت Embedding**: مدل ابری (سازگار با OpenAI) یا مدل محلی `sentence-transformers`
- 🗄️ **پایگاه داده برداری Qdrant** برای ذخیره و جست‌وجوی بردارهای معنایی
- 🔍 **بازیابی معنایی (Semantic Retrieval)** با آستانه‌ی امتیاز و `top_k` قابل تنظیم
- 🤖 **تولید پاسخ با LLM** از طریق API سازگار با OpenAI (به‌همراه پشتیبانی از **Streaming**)
- 📚 **استناد به منبع** (نام فایل + شماره صفحه + امتیاز شباهت) در پاسخ نهایی
- 💬 **رابط کاربری Streamlit** با چیدمان راست‌به‌چپ (RTL) و فارسی
- 👥 **تفکیک نقش‌ها**: کاربر عادی و مدیر سیستم
- 📤 **آپلود و ایندکس آنی اسناد** از داخل رابط کاربری
- 📊 **آمار و پنل دیباگ** (تعداد درخواست‌ها، میانگین زمان پاسخ، تعداد منابع)
- 🗃️ **مدیریت تاریخچه‌ی گفتگو** و اطلاعات کاربران با پایگاه‌داده JSON
- 🖥️ **رابط ترمینال** با نمایش گام‌به‌گام فرآیند RAG برای یادگیری عمیق‌تر

---

## 🧠 معماری و جریان کار RAG

```text
┌─────────────────────────────────────────────────────────────────────┐
│  فاز ایندکس‌سازی (Offline) — index_builder.py                        │
│                                                                     │
│  Data/*.pdf ──► استخراج متن ──► پاک‌سازی ──► Chunking ──► Embedding   │
│                                                              │      │
│                                                              ▼      │
│                                                    ┌──────────────┐ │
│                                                    │ Qdrant (DB)  │ │
│                                                    └──────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  فاز پرسش‌وپاسخ (Online) — main.py / streamlit_app.py                │
│                                                                     │
│  سوال کاربر ──► Embedding سوال ──► جست‌وجوی برداری در Qdrant          │
│        │                                     │                      │
│        │                                     ▼                      │
│        │                        قطعات مرتبط + امتیاز شباهت           │
│        │                                     │                      │
│        │                                     ▼                      │
│        │                    ساخت Context با استناد به منبع            │
│        │                                     │                      │
│        ▼                                     ▼                      │
│  ┌─────────────────────── LLM (API) ───────────────────────┐        │
│  │  System Prompt + Context + Question  ──►  پاسخ نهایی     │        │
│  └──────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

**قطعات اصلی سیستم:**

| قطعه | فایل | مسئولیت |
|---|---|---|
| پیکربندی | `settings/configs.py` | نگه‌داری تمام تنظیمات در یک `dataclass` و خواندن از `.env` |
| امبدینگ | `embeddings/embedder.py` | تبدیل متن به بردار (حالت `api` یا `local`) |
| پایگاه داده برداری | `vector_db/qdrant_db.py` | ساخت Collection، درج بردار و جست‌وجو |
| بازیاب | `retriever/retriever.py` | تبدیل سوال به بردار و یافتن نزدیک‌ترین قطعات |
| سازنده‌ی ایندکس | `index_builder.py` | پردازش PDF، Chunking و ایندکس‌گذاری |
| پوشش‌دهنده‌ی LLM | `agent/llm_wrapper.py` | ارتباط با API مدل زبانی |
| عامل RAG | `agent/rag_agent.py` | هماهنگی کل خط لوله و ساخت Context با منبع |
| پایگاه‌داده کاربران | `db_manager.py` | مدیریت مدیر، آمار، اسناد و تاریخچه‌ی چت |
| رابط ترمینال | `main.py` | نمایش گام‌به‌گام فرآیند RAG |
| رابط وب | `streamlit_app.py` | پنل چت، مدیریت اسناد و تنظیمات |

---

## 📁 ساختار پروژه

```text
rag-in-30-days/
├── agent/                  # عامل RAG و پوشش‌دهنده‌ی مدل زبانی
│   ├── llm_wrapper.py      # ارتباط با LLM (generate / stream)
│   ├── rag_agent.py        # هماهنگی بازیابی + تولید + ساخت Context
│   └── __init__.py
├── embeddings/             # تولید بردارهای معنایی
│   └── embedder.py
├── retriever/              # بازیابی قطعات مرتبط
│   └── retriever.py
├── vector_db/              # لایه‌ی ارتباط با Qdrant
│   └── qdrant_db.py
├── settings/               # پیکربندی مرکزی پروژه
│   └── configs.py
├── practical_oop/          # مثال‌های آموزشی برنامه‌نویسی شیءگرا (OOP)
│   ├── classes.py
│   ├── composision.py
│   ├── dataclass.py
│   └── static_methods.py
├── Data/                   # اسناد PDF ورودی (پیکره‌ی اسناد)
├── index_builder.py        # ساخت و مدیریت ایندکس برداری
├── main.py                 # رابط ترمینال (آموزشی، با نمایش جزئیات)
├── streamlit_app.py        # رابط وب کاربری
├── db_manager.py           # پایگاه‌داده JSON (کاربران، آمار، چت)
├── test.py                 # اسکریپت تست سریع امبدینگ
├── requirements.txt        # وابستگی‌های پروژه
├── .env                    # متغیرهای محیطی (کلیدهای API) — محرمانه
└── .gitignore
```

---

## 🛠️ تکنولوژی‌های استفاده‌شده

| حوزه | ابزار / کتابخانه |
|---|---|
| زبان | Python 3.10+ |
| رابط کاربری | Streamlit |
| پایگاه داده برداری | Qdrant (`qdrant-client`) |
| مدل امبدینگ | `sentence-transformers` / API سازگار با OpenAI |
| مدل زبانی | API سازگار با OpenAI (`openai` SDK) |
| پردازش PDF | `PyMuPDF` |
| پردازش متن فارسی | `hazm` |
| پیکربندی | `python-dotenv` + `dataclasses` |
| وب‌سرور (اختیاری) | `FastAPI` + `uvicorn` |
| پایگاه داده رابطه‌ای (اختیاری) | `SQLAlchemy` + `psycopg2-binary` |

---

## ⚙️ پیش‌نیازها

- **Python 3.10** یا بالاتر
- **Docker** (برای اجرای ساده‌ی Qdrant) — یا نصب دستی Qdrant
- **کلید API** برای سرویس امبدینگ و مدل زبانی (پیش‌فرض پروژه از `https://api.gapgpt.app/v1` استفاده می‌کند؛ هر سرویس سازگار با OpenAI نیز پشتیبانی می‌شود)

---

## 🚀 راه‌اندازی سریع

### ۱) دریافت سورس و ساخت محیط مجازی

```bash
git clone <repository-url>
cd rag-in-30-days

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### ۲) نصب وابستگی‌ها

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### ۳) راه‌اندازی Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

پس از اجرا، داشبورد Qdrant در آدرس `http://localhost:6333/dashboard` در دسترس است.

### ۴) تنظیم فایل `.env`

در ریشه‌ی پروژه فایل `.env` بسازید (یا از نمونه‌ی زیر استفاده کنید):

```env
# کلید سرویس امبدینگ
EMBEDDING_API_KEY=your-embedding-api-key

# کلید سرویس مدل زبانی
LLM_API_KEY=your-llm-api-key

# (اختیاری) اتصال به پایگاه داده رابطه‌ای
DATABASE_URL=postgresql://user:password@localhost:5432/rag_db
```

> 🔐 **هشدار امنیتی:** فایل `.env` حاوی اطلاعات محرمانه است و در `.gitignore` قرار دارد؛ هرگز آن را کامیت نکنید.

### ۵) ساخت ایندکس از اسناد

فایل‌های PDF خود را در پوشه‌ی `Data/` قرار دهید، سپس:

```bash
# ایندکس کردن همه‌ی PDFهای پوشه‌ی Data
python index_builder.py

# ایندکس کردن یک فایل مشخص
python index_builder.py --file Data/sample.pdf

# بازسازی کامل ایندکس
python index_builder.py --rebuild

# نمایش آمار ایندکس
python index_builder.py --stats
```

### ۶) اجرای برنامه

**رابط ترمینال (آموزشی، نمایش گام‌به‌گام فرآیند RAG):**

```bash
python main.py
```

**رابط وب Streamlit:**

```bash
streamlit run streamlit_app.py
```

سپس در مرورگر به آدرس `http://localhost:8501` مراجعه کنید.

---

## ⚙️ تنظیمات پروژه (`RAGConfig`)

تمام تنظیمات در `settings/configs.py` به‌صورت یک `dataclass` تعریف شده‌اند و مقادیر حساس از فایل `.env` خوانده می‌شوند:

| پارامتر | پیش‌فرض | توضیح |
|---|---|---|
| `embedder_type` | `api` | حالت امبدینگ: `api` یا `local` |
| `embedding_model` | `text-embedding-3-small` | مدل امبدینگ (۱۵۳۶ بُعد) |
| `local_embedding_model` | `intfloat/multilingual-e5-small` | مدل محلی در حالت `local` |
| `llm_model` | `gapgpt-qwen-3.5` | نام مدل زبانی |
| `qdrant_host` / `qdrant_port` | `localhost` / `6333` | آدرس Qdrant |
| `qdrant_collection` | `rag_documents` | نام Collection |
| `qdrant_vector_size` | `1536` | بُعد بردار (باید با مدل امبدینگ یکسان باشد) |
| `chunk_size` | `700` | اندازه‌ی هر قطعه (کاراکتر) |
| `overlap_sentences` | `2` | تعداد جملات همپوشانی بین قطعات |
| `default_top_k` | `3` | تعداد قطعات بازیابی‌شده |
| `min_score_threshold` | `0.35` | حداقل امتیاز شباهت برای پذیرش نتیجه |
| `temperature` | `0.3` | دمای خلاقیت مدل زبانی |
| `max_tokens` | `1024` | حداکثر توکن پاسخ |
| `data_dir` | `Data` | پوشه‌ی اسناد ورودی |

> 💡 مقادیر `top_k`، `min_score`، `temperature` و `max_tokens` از پنل مدیریت Streamlit نیز به‌صورت زنده قابل تغییر هستند.

---

## 🗂️ سرفصل‌های دوره و نگاشت به کد

دوره شامل **۸ فصل** و **۱۹ جلسه** است. هر بخش از سورس، خروجی عملی همان فصل‌هاست:

| فصل | عنوان | سورس متناظر |
|---|---|---|
| ۱ | مقدمه‌ای بر RAG و مفاهیم پایه | ساختار کلی پروژه و `agent/rag_agent.py` |
| ۲ | برنامه‌نویسی شیءگرای کاربردی (OOP) | `practical_oop/` و الگوهای شیءگرا در کل پروژه |
| ۳ | پیکربندی و راه‌اندازی (config و .env) | `settings/configs.py` + `.env` |
| ۴ | امبدینگ‌ها و پایگاه داده برداری | `embeddings/embedder.py` + `vector_db/qdrant_db.py` |
| ۵ | Index Builder (پیش‌پردازش و ساخت ایندکس) | `index_builder.py` |
| ۶ | بازیابی و ساخت Context | `retriever/retriever.py` + `agent/rag_agent.py` |
| ۷ | تولید پاسخ با مدل زبانی | `agent/llm_wrapper.py` |
| ۸ | رابط کاربری، مدیریت و استقرار | `streamlit_app.py` + `db_manager.py` + `main.py` |

---

## 🔐 نقش‌ها و ورود به سیستم

برنامه دو نقش دارد:

- **کاربر عادی:** پرسیدن سوال، مشاهده‌ی پاسخ همراه با منابع و پاک کردن گفتگو.
- **مدیر سیستم:** دسترسی به همه‌ی امکانات کاربر عادی به‌علاوه‌ی:
  - آپلود و ایندکس اسناد جدید
  - بازسازی کامل ایندکس
  - تنظیم زنده‌ی `top_k`، `min_score`، `temperature` و `max_tokens`
  - مشاهده‌ی آمار و پنل دیباگ

اطلاعات مدیر در فایل `app_data.json` نگه‌داری می‌شود.

> ⚠️ **هشدار:** اعتبارنامه‌ی پیش‌فرض مدیر `admin` / `admin123` است. حتماً پس از اولین اجرا رمز عبور را از طریق متد `update_admin_password` تغییر دهید. همچنین توجه داشته باشید که این فایل به‌دلیل نگه‌داری اطلاعات کاربران در `.gitignore` قرار دارد.

---

## 🧪 تست سریع

برای اطمینان از صحیح بودن اتصال به سرویس امبدینگ:

```bash
python test.py
```

این اسکریپت یک پرسش نمونه را به بردار تبدیل کرده و خروجی را چاپ می‌کند.

---

## 📌 نکات و رفع اشکال

| مشکل | راه‌حل |
|---|---|
| `Collection not found` | ابتدا `python index_builder.py` را اجرا کنید |
| `Qdrant: NOT REACHABLE` | مطمئن شوید کانتینر Qdrant در حال اجراست (`docker ps`) |
| خطای بُعد بردار | مقدار `qdrant_vector_size` باید با بُعد مدل امبدینگ یکسان باشد |
| پاسخی پیدا نمی‌شود | مقدار `min_score_threshold` را کاهش دهید یا `default_top_k` را افزایش دهید |
| خطای کلید API | صحت `EMBEDDING_API_KEY` و `LLM_API_KEY` در فایل `.env` را بررسی کنید |
| پاسخ‌های نامرتبط | مقدار `temperature` را کاهش دهید و کیفیت پیش‌پردازش متن را بررسی کنید |

---

## 🔗 لینک‌های مرتبط

- 🌐 وب‌سایت آکادمی کدیاد: [codeyad.com](https://codeyad.com)
- 🎓 صفحه‌ی دوره: [ساخت دستیار هوشمند اختصاصی در ۳۰ روز](https://codeyad.com/expert/build-custom-ai-assistant-in-30-days)

---

## 🎁 مزایای دوره

- ✅ گواهینامه پایان دوره
- ✅ وبینارهای آموزشی
- ✅ ضمانت بازگشت وجه
- ✅ گروه پشتیبانی تخصصی
- ✅ آپدیت همیشگی دوره

---

<p align="center">
ساخته‌شده با ❤️ در <b>آکادمی کدیاد</b>
</p>

</div>