"""
Central configuration for the RAG system.
Supports loading from .env and runtime updates.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()



@dataclass
class RAGConfig:
    """Main configuration class for the RAG system."""
    
    # ===== API & Endpoints =====
    embedding_api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    embedding_base_url: str = "https://api.gapgpt.app/v1"
    llm_base_url: str = "https://api.gapgpt.app/v1"
    
    # ===== Models =====
    embedding_model: str = "text-embedding-3-small"  # 1536 dimensions
    llm_model: str = "gapgpt-qwen-3.5"
    embedder_type: str = "api"  # "api" or "local"
    local_embedding_model: str = "intfloat/multilingual-e5-small"
    
    # ===== Paths =====
    data_dir: Path = Path("Data")
    # ===== Database =====
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://fardin:Fardin#poyan81@localhost:5432/rag_db"))
    
    # ===== Vector DB (Qdrant) =====
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "rag_documents"
    qdrant_vector_size: int = 1536  # Must match embedding model
    
    # ===== Chunking =====
    chunk_size: int = 700
    overlap_sentences: int = 2
    
    # ===== Retrieval =====
    default_top_k: int = 3
    min_score_threshold: float = 0.35
    
    # ===== LLM Generation =====
    temperature: float = 0.3
    max_tokens: int = 1024
    
    system_prompt: str = (
    "شما یک دستیار حرفه‌ای، خوش‌برخورد و مفید هستید که به سوالات کاربران پاسخ می‌دهید.\n\n"
    "ویژگی‌های شما:\n"
    "• رسمی اما صمیمی: پاسخ‌ها محترمانه و دوستانه هستند\n"
    "• دقیق و مفید: فقط بر اساس متن‌های ارائه شده پاسخ می‌دهید\n"
    "• تعاملی: اگر اطلاعات کافی نیست، راهنمایی می‌کنید که کاربر چه اطلاعات دیگری ارائه دهد\n"
    "• شفاف: حتماً ذکر می‌کنید که پاسخ بر اساس کدام بخش از اسناد است\n\n"
    "نحوه پاسخگویی:\n"
    "- اگر پاسخ در متن‌ها وجود دارد: پاسخ دقیق + ذکر منبع + پیشنهاد سوالات مرتبط\n"
    "- اگر پاسخ به طور کامل وجود ندارد: بهترین اطلاعات موجود را بدهید + صادقانه بگویید چه بخشی پیدا نشد\n"
    "- اگر هیچ اطلاعاتی وجود ندارد: صریح بگویید پیدا نشد + پیشنهاد تغییر عبارت جستجو\n\n"
    "همیشه پاسخ خود را به زبان فارسی روان و محترمانه بنویسید."
)

    rag_prompt_template: str = (
    "📄 **متن‌های مرتبط با سوال شما:**\n"
    "{context}\n\n"
    "❓ **سوال کاربر:**\n"
    "{query}\n\n"
    "📌 **دستورالعمل:**\n"
    "1. با دقت متن‌های بالا را بررسی کن\n"
    "2. به سوال کاربر به طور کامل و دقیق پاسخ بده\n"
    "3. اگر پاسخ در متن‌ها هست، حتماً به منبع (شماره و صفحه) ارجاع بده\n"
    "4. اگر پاسخ به طور کامل پیدا نشد، بهترین اطلاعات موجود را بده و صادقانه بگو چه بخشی کم است\n"
    "5. اگر هیچ اطلاعاتی پیدا نشد، مؤدبانه بگو و پیشنهاد بده سوال را با عبارت دیگری بپرسد\n"
    "6. در پایان، یک یا دو سوال مرتبط پیشنهاد بده که کاربر می‌تواند بپرسد\n\n"
    "✨ **پاسخ خود را به صورت زیر ساختار بده:**\n"
    "- ابتدا پاسخ اصلی به سوال\n"
    "- سپس ذکر منابع (با ذکر شماره و صفحه)\n"
    "- در انتها پیشنهاد سوالات مرتبط (در صورت مناسب بودن)"
    )



    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()


    def _validate(self):
        if self.embedder_type not in ("api", "local"):
            raise ValueError(f"embedder_type must be 'api' or 'local', got {self.embedder_type}")
        if not (0 < self.default_top_k <= 20):
            raise ValueError(f"default_top_k must be between 1 and 20, got {self.default_top_k}")
        if not (0.0 <= self.min_score_threshold <= 1.0):
            raise ValueError(f"min_score_threshold must be between 0 and 1, got {self.min_score_threshold}")
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError(f"temperature must be between 0 and 1, got {self.temperature}")
        if self.chunk_size < 100:
            raise ValueError(f"chunk_size must be at least 100, got {self.chunk_size}")


    def update(self, **kwargs):
        for key , value in kwargs.items():
            if hasattr(self,key):
                setattr(self, key , value)
            else:
                raise AttributeError(f"'{key}' is not a valid config attribute")


    def to_dict(self) -> dict:
        return {
            "embedding_api_key": "[HIDDEN]" if self.embedding_api_key else "",
            "llm_api_key": "[HIDDEN]" if self.llm_api_key else "",
            "embedding_base_url": self.embedding_base_url,
            "llm_base_url": self.llm_base_url,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "embedder_type": self.embedder_type,
            "local_embedding_model": self.local_embedding_model,
            "data_dir": str(self.data_dir),
            "qdrant_host": self.qdrant_host,
            "qdrant_port": self.qdrant_port,
            "qdrant_collection": self.qdrant_collection,
            "qdrant_vector_size": self.qdrant_vector_size,
            "chunk_size": self.chunk_size,
            "overlap_sentences": self.overlap_sentences,
            "default_top_k": self.default_top_k,
            "min_score_threshold": self.min_score_threshold,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "rag_prompt_template": self.rag_prompt_template
        }

        
    @classmethod
    def from_dict(cls, data: dict) -> "RAGConfig":
        if data.get('embedding_api_key') == "[HIDDEN]":
            data.pop("embedding_api_key", None)
        if data.get('llm_api_key') == "[HIDDEN]":
            data.pop("llm_api_key", None)
        if "data_dir" in data:
            data["data_dir"] = Path(data["data_dir"])

        return cls(**data)





























    
    # ===== Logging =====
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Perform validation on config values."""
        if self.embedder_type not in ("api", "local"):
            raise ValueError(f"embedder_type must be 'api' or 'local', got {self.embedder_type}")
        if not (0 < self.default_top_k <= 20):
            raise ValueError(f"default_top_k must be between 1 and 20, got {self.default_top_k}")
        if not (0.0 <= self.min_score_threshold <= 1.0):
            raise ValueError(f"min_score_threshold must be between 0 and 1, got {self.min_score_threshold}")
        if not (0.0 <= self.temperature <= 1.0):
            raise ValueError(f"temperature must be between 0 and 1, got {self.temperature}")
        if self.chunk_size < 100:
            raise ValueError(f"chunk_size must be at least 100, got {self.chunk_size}")
    
    def update(self, **kwargs):
        """Update config values at runtime (for UI)."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"'{key}' is not a valid config attribute")
        self._validate()
    

    def to_dict(self) -> dict:
        """Export config as dictionary (for saving/UI)."""
        return {
            "embedding_api_key": "[HIDDEN]" if self.embedding_api_key else "",
            "llm_api_key": "[HIDDEN]" if self.llm_api_key else "",
            "embedding_base_url": self.embedding_base_url,
            "llm_base_url": self.llm_base_url,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model,
            "embedder_type": self.embedder_type,
            "local_embedding_model": self.local_embedding_model,
            "data_dir": str(self.data_dir),
            "qdrant_host": self.qdrant_host,
            "qdrant_port": self.qdrant_port,
            "qdrant_collection": self.qdrant_collection,
            "qdrant_vector_size": self.qdrant_vector_size,
            "chunk_size": self.chunk_size,
            "overlap_sentences": self.overlap_sentences,
            "default_top_k": self.default_top_k,
            "min_score_threshold": self.min_score_threshold,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "rag_prompt_template": self.rag_prompt_template,
            "log_level": self.log_level,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RAGConfig":
        """Create config from dictionary (for loading saved presets)."""
        # Handle hidden API keys – if "[HIDDEN]" keep existing, else use new value
        if data.get("embedding_api_key") == "[HIDDEN]":
            data.pop("embedding_api_key", None)
        if data.get("llm_api_key") == "[HIDDEN]":
            data.pop("llm_api_key", None)
        # Convert data_dir string back to Path
        if "data_dir" in data:
            data["data_dir"] = Path(data["data_dir"])
        return cls(**data)


# Create a global default config instance (can be replaced by UI)
config = RAGConfig()