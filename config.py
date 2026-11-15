
from pathlib import Path


# ============================================================
# DIREKTORI PROJECT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

INDOBERT_DIR = MODEL_DIR / "indobert"

SENTENCE_BERT_DIR = MODEL_DIR / "sentence_bert"

OUTPUT_DIR = BASE_DIR / "outputs"

DATABASE_DIR = BASE_DIR / "database"


# ============================================================
# KONFIGURASI INDOBERT
# ============================================================

INDOBERT_BASE_MODEL = (
    "indobenchmark/indobert-base-p1"
)

INDOBERT_CHECKPOINT = "checkpoint-470"

NUM_LABELS = 3

LABEL2ID = {
    "negatif": 0,
    "netral": 1,
    "positif": 2
}

ID2LABEL = {
    0: "negatif",
    1: "netral",
    2: "positif"
}


# Panjang token saat fine-tuning
TRAINING_MAX_LENGTH = 64

# Mengikuti prediksi seluruh data pada notebook terpilih
INFERENCE_MAX_LENGTH = 128

# Batch size prediksi
INFERENCE_BATCH_SIZE = 16


# ============================================================
# KONFIGURASI SENTENCE-BERT
# ============================================================

SENTENCE_BERT_MODEL = (
    "paraphrase-multilingual-MiniLM-L12-v2"
)

EMBEDDING_DIMENSION = 384


# ============================================================
# PARAMETER ANALISIS NARASI
# ============================================================

INITIAL_SIMILARITY_THRESHOLD = 0.80

PAIR_SIMILARITY_THRESHOLD = 0.90

MAX_PAIR_TIME_MINUTES = 60

MIN_POSTS = 4

MIN_CROSS_ACCOUNT_PAIRS = 2

MIN_ACCOUNT_SIMILARITY = 0.90

MAX_ACCOUNT_TIME_MINUTES = 10

MIN_CANDIDATE_INDICATORS = 3


# ============================================================
# KOLOM DATASET
# ============================================================

REQUIRED_COLUMNS = [
    "id_str",
    "user_id_str",
    "created_at",
    "full_text"
]
