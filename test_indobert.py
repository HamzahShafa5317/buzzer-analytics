
import pandas as pd

from modules.sentiment_analysis import (
    load_indobert_model,
    predict_sentiment_dataframe
)

# ==========================================
# 1. DATA PENGUJIAN
# ==========================================

df_test = pd.DataFrame({

    "clean_text": [
        "program makan bergizi gratis sangat membantu siswa",
        "program mbg ini buruk dan tidak sesuai harapan",
        "pemerintah meluncurkan program makan bergizi gratis",
        ""
    ]

})

# ==========================================
# 2. LOAD MODEL
# ==========================================

print("Memuat model IndoBERT...")

tokenizer, model, device = load_indobert_model()

print("Model berhasil dimuat.")
print("Device:", device)

# ==========================================
# 3. PREDIKSI
# ==========================================

print("\nMenjalankan prediksi...")

hasil = predict_sentiment_dataframe(
    df=df_test,
    tokenizer=tokenizer,
    model=model,
    device=device,
    batch_size=2,
    max_length=128
)

# ==========================================
# 4. MENAMPILKAN HASIL
# ==========================================

print("\nHASIL PREDIKSI INDOBERT")
print("=" * 60)

print(
    hasil[
        [
            "clean_text",
            "sentimen_indobert",
            "confidence_indobert",
            "status_analisis_sentimen"
        ]
    ].to_string(index=False)
)

print("\nPengujian selesai.")
