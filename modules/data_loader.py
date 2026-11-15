
import pandas as pd
import re


# ==========================================
# NORMALISASI NAMA KOLOM
# ==========================================

def normalize_column_name(name):

    name = str(name).strip().lower()

    name = re.sub(
        r"[^a-z0-9]+",
        "_",
        name
    )

    return name.strip("_")


# ==========================================
# ALIAS NAMA KOLOM
# ==========================================

COLUMN_ALIASES = {

    "id_str": [
        "id_str",
        "id",
        "tweet_id",
        "post_id"
    ],

    "user_id_str": [
        "user_id_str",
        "user_id",
        "author_id",
        "userid"
    ],

    "created_at": [
        "created_at",
        "created_at_wib",
        "timestamp",
        "date"
    ],

    "full_text": [
        "full_text",
        "text",
        "tweet",
        "content"
    ],

    "clean_text": [
        "clean_text",
        "cleantext"
    ]
}


# ==========================================
# MENYERAGAMKAN KOLOM DATASET
# ==========================================

def standardize_columns(df):

    df = df.copy()

    normalized = {
        column: normalize_column_name(column)
        for column in df.columns
    }

    rename_mapping = {}

    for canonical, aliases in COLUMN_ALIASES.items():

        candidates = [
            column
            for column in df.columns
            if normalized[column] in aliases
        ]

        if candidates:

            # Prioritaskan nama kolom canonical
            selected = next(
                (
                    column for column in candidates
                    if normalized[column] == canonical
                ),
                candidates[0]
            )

            if selected != canonical:

                rename_mapping[selected] = canonical

    df = df.rename(
        columns=rename_mapping
    )

    # Jika hanya tersedia clean_text,
    # gunakan sebagai teks sumber sementara.
    if (
        "full_text" not in df.columns
        and "clean_text" in df.columns
    ):

        df["full_text"] = df["clean_text"]

    # Pertahankan ID sebagai string
    for column in [
        "id_str",
        "user_id_str"
    ]:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("")
                .astype("string")
                .str.strip()
            )

    return df


# ==========================================
# VALIDASI STRUKTUR DATASET
# ==========================================

def validate_dataset(df):

    required_columns = [
        "id_str",
        "user_id_str",
        "created_at",
        "full_text"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    return missing_columns


# ==========================================
# RINGKASAN DATASET
# ==========================================

def get_dataset_summary(df):

    summary = {

        "total_data": len(df),

        "jumlah_akun": (
            df["user_id_str"]
            .nunique()
        ),

        "teks_kosong": (
            df["full_text"]
            .fillna("")
            .str.strip()
            .eq("")
            .sum()
        ),

        "id_duplikat": (
            df["id_str"]
            .duplicated()
            .sum()
        )

    }

    return summary
