from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Models
    PLANNING_MODEL: str = "qwen2.5:7b-instruct"
    CODING_MODEL: str = "qwen2.5-coder:7b-instruct"
    VALIDATION_MODEL: str = "qwen2.5:7b-instruct"
    RESEARCH_MODEL: str = "qwen2.5:7b-instruct"

    # Temperatures
    PLANNING_TEMP: float = 0.3
    CODING_TEMP: float = 0.1
    VALIDATION_TEMP: float = 0.2
    RESEARCH_TEMP: float = 0.3

    # Execution
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MAX_RETRIES: int = 2
    CODE_EXEC_TIMEOUT: int = 15
    SERVER_STARTUP_TIMEOUT: int = 8
    LLM_CALL_TIMEOUT: int = 120

    # Paths
    USER_DATA_DIR: Path = Path(__file__).parent.parent / "user_data"
    OUTPUT_DIR: Path = Path(__file__).parent.parent / "outputs"
    DB_PATH: Path = Path("ai_intern_db.sqlite")

    # Planning
    USE_HIERARCHICAL_PLANNING: bool = True
    MAX_DECOMPOSITION_DEPTH: int = 3

    # Token management
    MAX_PROMPT_TOKENS: int = 4096

    # Logging
    LOG_LEVEL: str = "INFO"

    # llama-cpp-python settings (replacing Ollama)
    LLAMA_MODEL_PATH: str = r"E:\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    LLAMA_CODER_MODEL_PATH: str = r"E:\models\Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    LLAMA_N_GPU_LAYERS: int = 99    # Offload all layers to GPU
    LLAMA_N_CTX: int = 4096         # Context window
    LLAMA_N_THREADS: int = 8        # CPU threads for non-GPU ops
    LLAMA_VERBOSE: bool = False     # Suppress llama.cpp startup noise

    model_config = {
        "env_prefix": "AI_INTERN_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()

# Backwards compatibility
USE_HIERARCHICAL_PLANNING = settings.USE_HIERARCHICAL_PLANNING
