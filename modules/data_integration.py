
import re
import pandas as pd


# ============================================================
# 1. KONFIGURASI LABEL
# ============================================================

VALID_LABELS = [
    "negatif",
    "netral",
    "positif"
]

# Distribusi yang tercatat pada output data training
# notebook IndoBERT terpilih.
REFERENCE_TRAINING_DISTRIBUTION = {
    "negatif": 421,
    "netral": 286,
    "positif": 229
}


# ============================================================
# 2. NORMALISASI NAMA KOLOM
# ============================================================

def normalize_column_name(name):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(name).casefold()
    )


# ============================================================
# 3. NORMALISASI ID POSTING
# ID disimpan sebagai string, bukan angka.
# ============================================================

def normalize_id_series(series, source_name):

    values = (
        series.astype("string")
        .str.strip()
    )

    values = values.mask(
        values.eq(""),
        pd.NA
    )

    # Memperbaiki format string integer yang berakhiran .0
    # tanpa mengubah angka 19 digit menjadi integer.
    decimal_mask = values.str.fullmatch(
        r"\d+\.0+",
        na=False
    )

    values.loc[decimal_mask] = (
        values.loc[decimal_mask]
        .str.replace(
            r"\.0+$",
            "",
            regex=True
        )
    )

    # Menolak ID dalam format scientific notation
    # karena digit aslinya mungkin sudah kehilangan presisi.
    invalid_mask = (
        values.notna()
        & ~values.str.fullmatch(
            r"\d+",
            na=False
        )
    )

    if invalid_mask.any():

        examples = (
            values.loc[invalid_mask]
            .head(5)
            .tolist()
        )

        raise ValueError(
            f"ID pada {source_name} memiliki format tidak valid. "
            f"Contoh: {examples}. "
            "Pastikan ID disimpan sebagai teks di Excel."
        )

    if values.isna().any():

        raise ValueError(
            f"Ditemukan {int(values.isna().sum())} "
            f"ID kosong pada {source_name}."
        )

    return values


# ============================================================
# 4. MENYIAPKAN WORKBOOK ANOTASI
# ============================================================

def prepare_annotation_dataframe(df_annotation):

    df = df_annotation.copy()

    # Pemetaan nama kolom workbook anotasi
    aliases = {

        "idstr": "id_str",

        "relevansi": "relevansi",

        "kesepakatan": "kesepakatan",

        "sentimenpeneliti": "sentimen_peneliti",

        "validator": "sentimen_validator",

        "review": "review_validasi"

    }

    rename_mapping = {}

    for column in df.columns:

        normalized = normalize_column_name(
            column
        )

        if normalized in aliases:

            rename_mapping[column] = (
                aliases[normalized]
            )

    df = df.rename(
        columns=rename_mapping
    )

    # Kolom wajib
    required = [
        "id_str",
        "relevansi",
        "kesepakatan"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Kolom workbook anotasi tidak lengkap: {missing}"
        )

    # Normalisasi ID
    df["_id_key"] = normalize_id_series(
        df["id_str"],
        "workbook anotasi"
    )

    # Pastikan ID anotasi unik
    if df["_id_key"].duplicated().any():

        duplicated = (
            df.loc[
                df["_id_key"].duplicated(keep=False),
                "id_str"
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Ditemukan ID postingan duplikat pada workbook anotasi. "
            f"Contoh: {duplicated}"
        )

    # Normalisasi relevansi dan label
    df["relevansi"] = (
        df["relevansi"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["kesepakatan"] = (
        df["kesepakatan"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Validasi nilai relevansi
    invalid_relevance = ~df["relevansi"].isin(
        ["ya", "tidak"]
    )

    if invalid_relevance.any():

        raise ValueError(
            "Terdapat nilai relevansi selain 'ya' atau 'tidak'. "
            "Periksa workbook anotasi."
        )

    # Label hanya wajib valid untuk data relevan
    relevant_mask = df["relevansi"].eq("ya")

    invalid_label = (
        relevant_mask
        & ~df["kesepakatan"].isin(VALID_LABELS)
    )

    if invalid_label.any():

        raise ValueError(
            "Ditemukan label Kesepakatan tidak valid "
            "pada data relevan."
        )

    return df


# ============================================================
# 5. FUNGSI UTAMA INTEGRASI DATA
# ============================================================

def integrate_datasets(
    df_sentiment,
    df_annotation
):

    # --------------------------------------------------------
    # A. PERSIAPAN DATASET HASIL INDOBERT
    # --------------------------------------------------------

    base = df_sentiment.copy()

    required_base = [
        "id_str",
        "user_id_str",
        "created_at",
        "full_text",
        "clean_text",
        "sentimen_indobert"
    ]

    missing_base = [
        column
        for column in required_base
        if column not in base.columns
    ]

    if missing_base:

        raise ValueError(
            f"Kolom dataset hasil IndoBERT tidak lengkap: "
            f"{missing_base}"
        )

    # Hapus hasil integrasi sebelumnya jika ada.
    old_columns = [
        "sumber_label",
        "sentimen_final",
        "status_analisis_sentimen",
        "sentimen_analisis",
        "sentimen_indobert_raw",
        "relevansi_validasi",
        "label_final_validasi",
        "sentimen_peneliti_validasi",
        "sentimen_validator",
        "review_validasi"
    ]

    base = base.drop(
        columns=old_columns,
        errors="ignore"
    )

    # Hindari benturan dengan kolom anotasi lama
    # yang mungkin terdapat di dataset sumber.
    for column in list(base.columns):

        if normalize_column_name(column) in [
            "relevansi",
            "kesepakatan",
            "sentimenpeneliti",
            "validator",
            "review"
        ]:

            base = base.rename(
                columns={
                    column: f"{column}_sumber"
                }
            )

    # Normalisasi ID sumber
    base["_id_key"] = normalize_id_series(
        base["id_str"],
        "dataset hasil IndoBERT"
    )

    # ID dataset utama wajib unik
    if base["_id_key"].duplicated().any():

        raise ValueError(
            "ID postingan duplikat ditemukan "
            "pada dataset utama."
        )

    # --------------------------------------------------------
    # B. PERSIAPAN WORKBOOK ANOTASI
    # --------------------------------------------------------

    annotation = prepare_annotation_dataframe(
        df_annotation
    )

    # Kolom anotasi yang akan digabungkan
    annotation_columns = [
        "_id_key",
        "relevansi",
        "kesepakatan"
    ]

    optional_columns = [
        "sentimen_peneliti",
        "sentimen_validator",
        "review_validasi"
    ]

    for column in optional_columns:

        if column in annotation.columns:

            annotation_columns.append(
                column
            )

    annotation_merge = annotation[
        annotation_columns
    ].copy()

    # Menghindari nama kolom bertabrakan
    rename_annotation = {

        "relevansi":
            "annotation_relevansi",

        "kesepakatan":
            "annotation_kesepakatan",

        "sentimen_peneliti":
            "annotation_sentimen_peneliti",

        "sentimen_validator":
            "annotation_sentimen_validator",

        "review_validasi":
            "annotation_review"

    }

    annotation_merge = annotation_merge.rename(
        columns=rename_annotation
    )

    # --------------------------------------------------------
    # C. MERGE BERDASARKAN ID POSTING
    # --------------------------------------------------------

    result = base.merge(

        annotation_merge,

        on="_id_key",

        how="left",

        validate="one_to_one",

        indicator=True

    )

    matched_mask = result["_merge"].eq(
        "both"
    )

    unmatched_annotation = annotation.loc[
        ~annotation["_id_key"].isin(
            base["_id_key"]
        )
    ]

    if not unmatched_annotation.empty:

        examples = (
            unmatched_annotation["id_str"]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Ditemukan {len(unmatched_annotation)} "
            "ID anotasi yang tidak cocok dengan dataset utama. "
            f"Contoh ID: {examples}"
        )

    # --------------------------------------------------------
    # D. MEMISAHKAN KELOMPOK DATA
    # --------------------------------------------------------

    manual_mask = (
        matched_mask
        & result["annotation_relevansi"].eq("ya")
    )

    irrelevant_mask = (
        matched_mask
        & result["annotation_relevansi"].eq("tidak")
    )

    prediction_mask = ~matched_mask

    # --------------------------------------------------------
    # E. VALIDASI HASIL PREDIKSI INDOBERT
    # --------------------------------------------------------

    clean_text = (
        result["clean_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    raw_prediction = (
        result["sentimen_indobert"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Baris di luar anotasi yang memiliki teks tetapi
    # tidak memiliki prediksi harus diperiksa.
    missing_prediction = (
        prediction_mask
        & clean_text.ne("")
        & ~raw_prediction.isin(VALID_LABELS)
    )

    if missing_prediction.any():

        raise ValueError(
            f"Ditemukan {int(missing_prediction.sum())} "
            "unggahan non-anotasi yang belum memiliki "
            "prediksi IndoBERT. Jalankan prediksi terlebih dahulu."
        )

    # Simpan prediksi asli untuk keperluan audit.
    result["sentimen_indobert_raw"] = (
        raw_prediction
    )

    # --------------------------------------------------------
    # F. MENYIAPKAN KOLOM OUTPUT
    # --------------------------------------------------------

    result["sentimen_final"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string"
    )

    result["sumber_label"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string"
    )

    result["status_analisis_sentimen"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string"
    )

    result["sentimen_indobert"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="string"
    )

    # --------------------------------------------------------
    # G. LABEL MANUAL YANG RELEVAN
    # --------------------------------------------------------

    result.loc[
        manual_mask,
        "sentimen_final"
    ] = result.loc[
        manual_mask,
        "annotation_kesepakatan"
    ]

    result.loc[
        manual_mask,
        "sumber_label"
    ] = "manual_validasi"

    result.loc[
        manual_mask,
        "status_analisis_sentimen"
    ] = "label_manual_relevan"

    # --------------------------------------------------------
    # H. PREDIKSI INDOBERT DI LUAR ANOTASI
    # --------------------------------------------------------

    valid_prediction_mask = (
        prediction_mask
        & raw_prediction.isin(VALID_LABELS)
    )

    result.loc[
        valid_prediction_mask,
        "sentimen_indobert"
    ] = raw_prediction.loc[
        valid_prediction_mask
    ]

    result.loc[
        valid_prediction_mask,
        "sentimen_final"
    ] = raw_prediction.loc[
        valid_prediction_mask
    ]

    result.loc[
        prediction_mask,
        "sumber_label"
    ] = "IndoBERT"

    result.loc[
        valid_prediction_mask,
        "status_analisis_sentimen"
    ] = "hasil_prediksi_indobert"

    # --------------------------------------------------------
    # I. DATA TIDAK RELEVAN
    # --------------------------------------------------------

    result.loc[
        irrelevant_mask,
        "sumber_label"
    ] = "tidak_relevan"

    result.loc[
        irrelevant_mask,
        "status_analisis_sentimen"
    ] = "dikeluarkan_tidak_relevan"

    # --------------------------------------------------------
    # J. PENANGANAN TEKS KOSONG
    # --------------------------------------------------------

    empty_prediction_mask = (
        prediction_mask
        & clean_text.eq("")
    )

    result.loc[
        empty_prediction_mask,
        "status_analisis_sentimen"
    ] = "teks_kosong_tidak_diprediksi"

    result.loc[
        empty_prediction_mask,
        "sentimen_final"
    ] = pd.NA

    # --------------------------------------------------------
    # K. MENYIMPAN INFORMASI ANOTASI
    # --------------------------------------------------------

    result["relevansi_validasi"] = (
        result["annotation_relevansi"]
        .astype("string")
    )

    result["label_final_validasi"] = (
        result["annotation_kesepakatan"]
        .astype("string")
    )

    for source_column, target_column in [

        (
            "annotation_sentimen_peneliti",
            "sentimen_peneliti_validasi"
        ),

        (
            "annotation_sentimen_validator",
            "sentimen_validator"
        ),

        (
            "annotation_review",
            "review_validasi"
        )

    ]:

        if source_column in result.columns:

            result[target_column] = (
                result[source_column]
            )

        else:

            result[target_column] = pd.NA

    # --------------------------------------------------------
    # L. SENTIMEN UNTUK ANALISIS LANJUTAN
    # --------------------------------------------------------

    result["sentimen_analisis"] = (
        result["sentimen_final"]
    )

    # Bersihkan kolom internal
    result = result.drop(
        columns=[
            "_id_key",
            "_merge",
            "annotation_relevansi",
            "annotation_kesepakatan",
            "annotation_sentimen_peneliti",
            "annotation_sentimen_validator",
            "annotation_review"
        ],
        errors="ignore"
    )

    result["clean_text"] = clean_text

    # --------------------------------------------------------
    # M. PENYARINGAN DATA UNTUK SENTENCE-BERT
    # --------------------------------------------------------

    irrelevant_final = (
        result["status_analisis_sentimen"]
        .eq("dikeluarkan_tidak_relevan")
    )

    empty_text_mask = (
        result["clean_text"].eq("")
    )

    analysis_mask = (
        ~irrelevant_final
        & ~empty_text_mask
    )

    analysis_df = (
        result.loc[analysis_mask]
        .copy()
        .reset_index(drop=True)
    )

    if analysis_df.empty:

        raise ValueError(
            "Tidak terdapat data yang dapat dianalisis."
        )

    # Pastikan seluruh data analisis memiliki label
    if not analysis_df[
        "sentimen_analisis"
    ].isin(VALID_LABELS).all():

        raise ValueError(
            "Terdapat data analisis yang belum memiliki "
            "sentimen final."
        )

    # Validasi ID unggahan dan ID akun
    if not analysis_df["id_str"].is_unique:

        raise ValueError(
            "ID unggahan tidak unik pada data analisis."
        )

    user_ids = (
        analysis_df["user_id_str"]
        .astype("string")
        .str.strip()
    )

    if user_ids.isna().any() or user_ids.eq("").any():

        raise ValueError(
            "Terdapat ID akun kosong pada data analisis."
        )

    analysis_df["user_id_str"] = user_ids

    # --------------------------------------------------------
    # N. NORMALISASI WAKTU
    # --------------------------------------------------------

    created_utc = pd.to_datetime(
        analysis_df["created_at"],
        utc=True,
        errors="coerce"
    )

    if created_utc.isna().any():

        raise ValueError(
            "Terdapat waktu publikasi tidak valid "
            "pada data analisis."
        )

    analysis_df["created_at_utc"] = (
        created_utc
    )

    analysis_df["created_at_wib"] = (
        created_utc.dt.tz_convert(
            "Asia/Jakarta"
        )
    )

    # --------------------------------------------------------
    # O. RINGKASAN HASIL INTEGRASI
    # --------------------------------------------------------

    actual_manual_distribution = (
        result.loc[
            manual_mask,
            "sentimen_final"
        ]
        .value_counts()
        .reindex(
            VALID_LABELS,
            fill_value=0
        )
        .astype(int)
        .to_dict()
    )

    analysis_distribution = (
        analysis_df["sentimen_analisis"]
        .value_counts()
        .reindex(
            VALID_LABELS,
            fill_value=0
        )
        .astype(int)
        .to_dict()
    )

    number_empty = int(
        (
            ~irrelevant_final
            & empty_text_mask
        ).sum()
    )

    summary = {

        "jumlah_dataset_utama": len(result),

        "jumlah_anotasi_cocok": int(
            matched_mask.sum()
        ),

        "jumlah_manual_relevan": int(
            manual_mask.sum()
        ),

        "jumlah_tidak_relevan": int(
            irrelevant_mask.sum()
        ),

        "jumlah_data_diluar_anotasi": int(
            prediction_mask.sum()
        ),

        "jumlah_prediksi_valid": int(
            valid_prediction_mask.sum()
        ),

        "jumlah_teks_kosong_dikeluarkan": (
            number_empty
        ),

        "jumlah_data_analisis": len(
            analysis_df
        ),

        "jumlah_akun_analisis": (
            analysis_df["user_id_str"].nunique()
        ),

        "distribusi_label_manual": (
            actual_manual_distribution
        ),

        "distribusi_sentimen_analisis": (
            analysis_distribution
        ),

        "distribusi_training_notebook": (
            REFERENCE_TRAINING_DISTRIBUTION.copy()
        ),

        "sesuai_distribusi_training": (
            actual_manual_distribution
            == REFERENCE_TRAINING_DISTRIBUTION
        )

    }

    return result, analysis_df, summary
