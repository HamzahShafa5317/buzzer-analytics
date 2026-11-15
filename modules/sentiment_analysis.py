
import pandas as pd
import numpy as np
import torch

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from config import (
    INDOBERT_DIR,
    ID2LABEL,
    NUM_LABELS,
    INFERENCE_MAX_LENGTH,
    INFERENCE_BATCH_SIZE
)


# ============================================================
# 1. MEMUAT MODEL INDOBERT
# ============================================================

def load_indobert_model(model_dir=None):

    if model_dir is None:
        model_dir = INDOBERT_DIR

    model_dir = Path(model_dir)

    # Validasi folder model
    if not model_dir.exists():

        raise FileNotFoundError(
            f"Folder model IndoBERT tidak ditemukan: {model_dir}"
        )

    if not (model_dir / "config.json").exists():

        raise FileNotFoundError(
            "config.json tidak ditemukan. "
            "Periksa isi folder model/indobert."
        )

    # Memuat tokenizer lokal
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True
    )

    # Memuat bobot model hasil fine-tuning
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir),
        local_files_only=True
    )

    # Validasi jumlah kelas
    if model.config.num_labels != NUM_LABELS:

        raise ValueError(
            f"Jumlah label model adalah {model.config.num_labels}, "
            f"seharusnya {NUM_LABELS}."
        )

    # Validasi mapping label
    loaded_mapping = {
        int(key): str(value).lower()
        for key, value in model.config.id2label.items()
    }

    if loaded_mapping != ID2LABEL:

        raise ValueError(
            "Mapping label model tidak sesuai "
            "dengan konfigurasi penelitian."
        )

    # Menentukan perangkat komputasi
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)

    # Mode evaluasi
    model.eval()

    return tokenizer, model, device


# ============================================================
# 2. FUNGSI PREDIKSI SENTIMEN
# ============================================================

def predict_sentiment_dataframe(
    df,
    tokenizer,
    model,
    device=None,
    batch_size=INFERENCE_BATCH_SIZE,
    max_length=INFERENCE_MAX_LENGTH,
    progress_callback=None
):

    if "clean_text" not in df.columns:

        raise ValueError(
            "Kolom clean_text tidak ditemukan."
        )

    if batch_size < 1:

        raise ValueError(
            "Batch size harus lebih besar dari nol."
        )

    if max_length < 1:

        raise ValueError(
            "Max length harus lebih besar dari nol."
        )

    # Salinan dataset
    result = df.copy().reset_index(drop=True)

    total_data = len(result)

    # --------------------------------------------------------
    # 1. MENYIAPKAN TEKS
    # --------------------------------------------------------

    texts = []

    for value in result["clean_text"]:

        if pd.isna(value):
            texts.append("")
        else:
            texts.append(str(value).strip())

    # Indeks teks yang tidak kosong
    valid_indices = [
        i
        for i, text in enumerate(texts)
        if text.strip()
    ]

    valid_texts = [
        texts[i]
        for i in valid_indices
    ]

    # --------------------------------------------------------
    # 2. MENYIAPKAN OUTPUT
    # --------------------------------------------------------

    predictions = [
        pd.NA
        for _ in range(total_data)
    ]

    confidences = np.full(
        total_data,
        np.nan
    )

    probabilities = np.full(
        (total_data, NUM_LABELS),
        np.nan
    )

    statuses = [
        "teks_kosong_tidak_diprediksi"
        for _ in range(total_data)
    ]

    # --------------------------------------------------------
    # 3. MENENTUKAN DEVICE
    # --------------------------------------------------------

    if device is None:

        device = next(
            model.parameters()
        ).device

    # --------------------------------------------------------
    # 4. BATCH PREDICTION
    # --------------------------------------------------------

    model.eval()

    total_valid = len(valid_texts)

    with torch.inference_mode():

        for start in range(
            0,
            total_valid,
            batch_size
        ):

            batch_texts = valid_texts[
                start:start + batch_size
            ]

            # Tokenisasi
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            # Forward pass
            outputs = model(
                **encoded
            )

            logits = outputs.logits

            if logits.shape[1] != NUM_LABELS:

                raise ValueError(
                    "Jumlah output model tidak sesuai."
                )

            # Probabilitas softmax
            probs = torch.softmax(
                logits.float(),
                dim=1
            ).cpu().numpy()

            # Prediksi kelas
            pred_ids = np.argmax(
                probs,
                axis=1
            )

            confidence = np.max(
                probs,
                axis=1
            )

            # ------------------------------------------------
            # SIMPAN HASIL SESUAI URUTAN DATA
            # ------------------------------------------------

            for j, pred_id in enumerate(pred_ids):

                original_index = valid_indices[
                    start + j
                ]

                predictions[original_index] = (
                    ID2LABEL[int(pred_id)]
                )

                confidences[original_index] = (
                    float(confidence[j])
                )

                probabilities[original_index] = (
                    probs[j]
                )

                statuses[original_index] = (
                    "hasil_prediksi_indobert"
                )

            # ------------------------------------------------
            # UPDATE PROGRESS
            # ------------------------------------------------

            processed = min(
                start + batch_size,
                total_valid
            )

            if progress_callback is not None:

                progress_callback(
                    processed,
                    total_valid
                )

    # --------------------------------------------------------
    # 5. MENAMBAHKAN HASIL KE DATAFRAME
    # --------------------------------------------------------

    result["sentimen_indobert"] = pd.Series(
        predictions,
        dtype="string"
    )

    result["confidence_indobert"] = (
        confidences
    )

    result["probabilitas_negatif"] = (
        probabilities[:, 0]
    )

    result["probabilitas_netral"] = (
        probabilities[:, 1]
    )

    result["probabilitas_positif"] = (
        probabilities[:, 2]
    )

    result["status_analisis_sentimen"] = (
        statuses
    )

    return result
