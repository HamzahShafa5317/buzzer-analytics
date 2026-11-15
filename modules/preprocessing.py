
import re
import pandas as pd


# ============================================================
# 1. POLA REGEX EMOJI
# Mengikuti pola pada notebook Preprocessing_FINAL
# ============================================================

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]",
    flags=re.UNICODE
)


# ============================================================
# 2. FUNGSI CASE FOLDING
# Mengubah seluruh huruf menjadi lowercase
# ============================================================

def case_folding(text):

    if pd.isna(text):
        return ""

    return str(text).lower()


# ============================================================
# 3. FUNGSI CLEANING TEXT
# Menghapus newline, tab, dan karakter kontrol
# ============================================================

def cleaning_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Menghapus newline, carriage return, dan tab
    text = re.sub(
        r"[\r\n\t]+",
        " ",
        text
    )

    # Menghapus karakter kontrol
    text = re.sub(
        r"[\x00-\x1F\x7F]",
        " ",
        text
    )

    # Merapikan spasi
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 4. NORMALISASI REPEATED CHARACTER
# Maksimal 2 karakter berulang
# ============================================================

def normalize_repeated_character(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Menormalkan tanda baca berulang
    # !!!! -> !
    # ???? -> ?
    text = re.sub(
        r'([.!?,;:])\1+',
        r'\1',
        text
    )

    # Maksimal 2 karakter alfanumerik berulang
    # baguuuus -> baguus
    # bangettt -> bangett
    text = re.sub(
        r'([a-zA-Z0-9])\1{2,}',
        r'\1\1',
        text
    )

    # Merapikan spasi
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# ============================================================
# 5. REMOVE URL
# Menghapus URL dari teks
# ============================================================

def remove_url(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Menghapus http dan https
    text = re.sub(
        r'https?://\S+',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Menghapus www
    text = re.sub(
        r'www\.\S+',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Menghapus t.co
    text = re.sub(
        r'(?<!\w)t\.co/\S+',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # Merapikan spasi
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# ============================================================
# 6. REMOVE MENTION
# Menghapus mention @username
# ============================================================

def remove_mention(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = re.sub(
        r'@\w+',
        ' ',
        text
    )

    # Merapikan spasi
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# ============================================================
# 7. HASHTAG PROCESSING
# Menghapus simbol # tetapi mempertahankan isi hashtag
# ============================================================

def process_hashtag(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # #mbg -> mbg
    # #mbgIndonesia -> mbgIndonesia
    text = re.sub(
        r'#(\w+)',
        r'\1',
        text
    )

    # Merapikan spasi
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# ============================================================
# 8. REMOVE EMOJI
# Menghapus emoji dan karakter variasinya
# ============================================================

def remove_emoji(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = EMOJI_PATTERN.sub(
        ' ',
        text
    )

    # Menghapus variation selector dan zero-width joiner
    text = re.sub(
        r'[\uFE0F\u200D]',
        ' ',
        text
    )

    # Merapikan spasi
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# ============================================================
# 9. PREPROCESSING UTAMA
# Menghasilkan clean_text dan tahapan preprocessing
# ============================================================

def preprocess_text_dataframe(df):

    df = df.copy()

    if "full_text" not in df.columns:

        raise ValueError(
            "Kolom full_text tidak ditemukan. "
            "Periksa dataset yang diunggah."
        )

    # 1. Case folding
    df["case_folding"] = (
        df["full_text"]
        .apply(case_folding)
    )

    # 2. Cleaning text
    df["cleaning"] = (
        df["case_folding"]
        .apply(cleaning_text)
    )

    # 3. Normalisasi repeated character
    df["normalisasi_repeated"] = (
        df["cleaning"]
        .apply(normalize_repeated_character)
    )

    # 4. Remove URL
    df["remove_url"] = (
        df["normalisasi_repeated"]
        .apply(remove_url)
    )

    # 5. Remove mention
    df["remove_mention"] = (
        df["remove_url"]
        .apply(remove_mention)
    )

    # 6. Hashtag processing
    df["hashtag_processing"] = (
        df["remove_mention"]
        .apply(process_hashtag)
    )

    # 7. Remove emoji
    df["remove_emoji"] = (
        df["hashtag_processing"]
        .apply(remove_emoji)
    )

    # 8. Hasil akhir preprocessing
    df["clean_text"] = (
        df["remove_emoji"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# 10. REMOVE DUPLICATE BERDASARKAN ID POSTINGAN
# Hanya menghapus ID yang berulang
# ============================================================

def remove_duplicate_post(df):

    df = df.copy()

    if "id_str" not in df.columns:

        raise ValueError(
            "Kolom id_str tidak ditemukan."
        )

    id_postingan = (
        df["id_str"]
        .astype("string")
        .str.strip()
    )

    # ID kosong tidak dianggap duplikat
    id_valid = (
        id_postingan.notna()
        & id_postingan.ne("")
    )

    # Menandai ID postingan yang berulang
    duplicate_mask = (
        id_valid
        & id_postingan.duplicated(keep="first")
    )

    jumlah_sebelum = len(df)

    jumlah_duplicate = int(
        duplicate_mask.sum()
    )

    jumlah_id_kosong = int(
        (~id_valid).sum()
    )

    # Menghapus duplikasi berdasarkan ID saja
    df = (
        df.loc[~duplicate_mask]
        .copy()
        .reset_index(drop=True)
    )

    info = {
        "jumlah_sebelum": jumlah_sebelum,
        "jumlah_duplicate": jumlah_duplicate,
        "jumlah_id_kosong": jumlah_id_kosong,
        "jumlah_sesudah": len(df)
    }

    return df, info


# ============================================================
# 11. VALIDASI HASIL PREPROCESSING
# ============================================================

def validate_preprocessing(df):

    if "clean_text" not in df.columns:

        raise ValueError(
            "Kolom clean_text belum tersedia."
        )

    if "id_str" not in df.columns:

        raise ValueError(
            "Kolom id_str tidak ditemukan."
        )

    clean_text = (
        df["clean_text"]
        .fillna("")
        .astype(str)
    )

    hasil = {

        "total_data": len(df),

        "teks_kosong": int(
            clean_text.str.strip().eq("").sum()
        ),

        "sisa_url": int(
            clean_text.str.contains(
                r'https?://\S+|www\.\S+|(?<!\w)t\.co/\S+',
                regex=True,
                na=False
            ).sum()
        ),

        "sisa_mention": int(
            clean_text.str.contains(
                r'@\w+',
                regex=True,
                na=False
            ).sum()
        ),

        "sisa_hashtag": int(
            clean_text.str.contains(
                r'#\w+',
                regex=True,
                na=False
            ).sum()
        ),

        "sisa_repeated_character": int(
            clean_text.apply(
                lambda x: bool(
                    re.search(
                        r'([a-zA-Z0-9])\1{2,}',
                        x
                    )
                )
            ).sum()
        ),

        "sisa_punctuation": int(
            clean_text.apply(
                lambda x: bool(
                    re.search(
                        r'([.!?,;:])\1+',
                        x
                    )
                )
            ).sum()
        )

    }

    return hasil


# ============================================================
# 12. FUNGSI PREPROCESSING LENGKAP
# ============================================================

def run_preprocessing(df):

    # Simpan jumlah data awal
    jumlah_awal = len(df)

    # Jalankan preprocessing teks
    df_processed = preprocess_text_dataframe(df)

    # Hapus duplikasi berdasarkan ID posting
    df_processed, duplicate_info = (
        remove_duplicate_post(df_processed)
    )

    # Validasi hasil akhir
    validation_info = validate_preprocessing(
        df_processed
    )

    # Statistik akun unik
    jumlah_akun = 0

    if "user_id_str" in df_processed.columns:

        akun = (
            df_processed["user_id_str"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        jumlah_akun = akun[
            akun.ne("")
        ].nunique()

    summary = {

        "jumlah_data_awal": jumlah_awal,

        "jumlah_data_duplikat": (
            duplicate_info["jumlah_duplicate"]
        ),

        "jumlah_id_kosong": (
            duplicate_info["jumlah_id_kosong"]
        ),

        "jumlah_data_akhir": len(df_processed),

        "jumlah_akun_unik": jumlah_akun,

        "validasi": validation_info

    }

    return df_processed, summary
