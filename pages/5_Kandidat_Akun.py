
import io

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

from modules.narrative_analysis import (
    MIN_POSTS,
    MIN_CROSS_ACCOUNT_PAIRS,
    MIN_ACCOUNT_SIMILARITY,
    MAX_ACCOUNT_TIME_MINUTES,
    MIN_CANDIDATE_INDICATORS,
    LABEL_SENTIMEN
)


# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Identifikasi Kandidat Akun",
    page_icon="Users",
    layout="wide"
)


# ============================================================
# 2. FUNGSI PENDUKUNG
# ============================================================

def excel_safe_dataframe(df):

    output = df.copy()

    for column in output.columns:

        if isinstance(
            output[column].dtype,
            pd.DatetimeTZDtype
        ):

            output[column] = (
                output[column]
                .dt.strftime("%Y-%m-%d %H:%M:%S %Z")
            )

    return output


def format_number(value, decimals=2):

    if pd.isna(value):
        return "-"

    return f"{float(value):,.{decimals}f}"


def get_indicator_status(value):

    return "Terpenuhi" if int(value) == 1 else "Tidak terpenuhi"


def get_candidate_display(user_id, profile_column=None, posts=None):

    if profile_column and posts is not None:

        profile_values = posts[
            profile_column
        ].dropna()

        profile_values = (
            profile_values.astype(str).str.strip()
        )

        profile_values = profile_values[
            profile_values.ne("")
        ]

        if not profile_values.empty:

            return (
                f"{profile_values.iloc[0]} "
                f"({user_id})"
            )

    return str(user_id)


# ============================================================
# 3. HEADER
# ============================================================

st.title(
    "Identifikasi Kandidat Akun"
)

st.caption(
    "Identifikasi akun berdasarkan indikator aktivitas "
    "komunikasi terkoordinasi dengan informasi pendukung "
    "berupa analisis sentimen."
)

st.divider()


# ============================================================
# 4. VALIDASI HASIL ANALISIS NARASI
# ============================================================

if "narrative_results" not in st.session_state:

    st.warning(
        """
        Hasil analisis narasi belum tersedia.

        Silakan jalankan Analisis Sentence-BERT terlebih dahulu
        melalui halaman Analisis Narasi.
        """
    )

    st.stop()


results = st.session_state[
    "narrative_results"
]


# Mengambil dataframe hasil analisis

candidates = results[
    "df_kandidat_final"
].copy()

indicators = results[
    "df_indikator"
].copy()

df_analysis = results[
    "df_analysis"
].copy()

df_pairs = results[
    "df_sentimen_pasangan"
].copy()

df_relations = results[
    "df_hubungan_akun"
].copy()

df_account_map = results[
    "df_pemetaan_akun"
].copy()

summary = results[
    "summary"
]


# ============================================================
# 5. VALIDASI STRUKTUR DATA
# ============================================================

required_candidate_columns = [

    "user_id",
    "jumlah_posting",
    "jumlah_pasangan_lintas_akun",
    "rata_similarity",
    "rata_selisih_waktu_menit",
    "indikator_posting",
    "indikator_lintas_akun",
    "indikator_similarity",
    "indikator_temporal",
    "jumlah_indikator",
    "sentimen_dominan"

]

missing_columns = [

    column
    for column in required_candidate_columns
    if column not in candidates.columns

]

if missing_columns:

    st.error(
        f"Kolom kandidat tidak ditemukan: {missing_columns}"
    )

    st.stop()


# Pastikan ID dalam bentuk string

candidates["user_id"] = (
    candidates["user_id"].astype(str)
)

df_analysis["user_id_str"] = (
    df_analysis["user_id_str"].astype(str)
)


# ============================================================
# 6. RINGKASAN IDENTIFIKASI
# ============================================================

st.header(
    "Ringkasan Identifikasi Kandidat"
)

total_accounts = summary.get(
    "jumlah_akun_dievaluasi",
    len(indicators)
)

total_candidates = len(
    candidates
)

total_relations = len(
    df_relations
)

candidate_percentage = (

    total_candidates / total_accounts * 100

    if total_accounts > 0

    else 0

)


k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Akun Dievaluasi",
    f"{total_accounts:,}"
)

k2.metric(
    "Kandidat Akun",
    f"{total_candidates:,}"
)

k3.metric(
    "Persentase Kandidat",
    f"{candidate_percentage:.2f}%"
)

k4.metric(
    "Hubungan Akun",
    f"{total_relations:,}"
)


# ============================================================
# 7. PENJELASAN INDIKATOR
# ============================================================

st.divider()

st.subheader(
    "Kriteria Identifikasi Kandidat"
)

st.write(
    """
    Sistem menggunakan empat indikator aktivitas akun.
    Akun ditetapkan sebagai kandidat apabila memenuhi
    minimal tiga dari empat indikator tersebut.
    """
)


indicator_reference = pd.DataFrame({

    "Indikator": [

        "Jumlah posting",

        "Pasangan lintas akun",

        "Rata-rata similarity",

        "Rata-rata selisih waktu"

    ],

    "Kriteria": [

        f"≥ {MIN_POSTS} posting",

        f"≥ {MIN_CROSS_ACCOUNT_PAIRS} pasangan",

        f"≥ {MIN_ACCOUNT_SIMILARITY:.2f}",

        f"≤ {MAX_ACCOUNT_TIME_MINUTES} menit"

    ]

})

st.dataframe(
    indicator_reference,
    use_container_width=True,
    hide_index=True
)

st.info(
    f"Seleksi kandidat menggunakan minimal "
    f"{MIN_CANDIDATE_INDICATORS} dari 4 indikator. "
    "Sentimen tidak digunakan sebagai indikator seleksi."
)


# ============================================================
# 8. FILTER KANDIDAT
# ============================================================

st.divider()

st.header(
    "Daftar Kandidat Akun"
)

filter_col1, filter_col2 = st.columns(2)

with filter_col1:

    selected_indicator = st.selectbox(

        "Filter jumlah indikator",

        options=[
            "Semua kandidat",
            "3 indikator",
            "4 indikator"
        ],

        index=0

    )


with filter_col2:

    search_account = st.text_input(

        "Cari ID akun",

        placeholder="Masukkan ID akun..."

    )


filtered_candidates = candidates.copy()


if selected_indicator == "3 indikator":

    filtered_candidates = filtered_candidates.loc[
        filtered_candidates["jumlah_indikator"] == 3
    ]


elif selected_indicator == "4 indikator":

    filtered_candidates = filtered_candidates.loc[
        filtered_candidates["jumlah_indikator"] == 4
    ]


if search_account.strip():

    filtered_candidates = filtered_candidates.loc[

        filtered_candidates["user_id"]
        .str.contains(
            search_account.strip(),
            case=False,
            na=False
        )

    ]


# ============================================================
# 9. TABEL KANDIDAT
# ============================================================

candidate_display_columns = [

    "user_id",

    "jumlah_posting",

    "jumlah_pasangan_lintas_akun",

    "rata_similarity",

    "rata_selisih_waktu_menit",

    "indikator_posting",

    "indikator_lintas_akun",

    "indikator_similarity",

    "indikator_temporal",

    "jumlah_indikator",

    "sentimen_dominan"

]

candidate_display_columns = [

    column
    for column in candidate_display_columns
    if column in filtered_candidates.columns

]


candidate_display = filtered_candidates[
    candidate_display_columns
].copy()


# Mengubah indikator 1/0 menjadi teks

indicator_label_map = {

    "indikator_posting":
        "Posting memenuhi",

    "indikator_lintas_akun":
        "Lintas akun memenuhi",

    "indikator_similarity":
        "Similarity memenuhi",

    "indikator_temporal":
        "Temporal memenuhi"

}


for column in indicator_label_map:

    if column in candidate_display.columns:

        candidate_display[column] = (
            candidate_display[column]
            .apply(get_indicator_status)
        )


candidate_display = candidate_display.rename(

    columns={

        "user_id":
            "ID Akun",

        "jumlah_posting":
            "Jumlah Posting",

        "jumlah_pasangan_lintas_akun":
            "Jumlah Pasangan",

        "rata_similarity":
            "Rata-rata Similarity",

        "rata_selisih_waktu_menit":
            "Rata-rata Selisih Waktu (menit)",

        "indikator_posting":
            "Indikator Posting",

        "indikator_lintas_akun":
            "Indikator Lintas Akun",

        "indikator_similarity":
            "Indikator Similarity",

        "indikator_temporal":
            "Indikator Temporal",

        "jumlah_indikator":
            "Jumlah Indikator",

        "sentimen_dominan":
            "Sentimen Dominan"

    }

)


st.metric(
    "Jumlah kandidat sesuai filter",
    f"{len(filtered_candidates):,}"
)


if filtered_candidates.empty:

    st.info(
        "Tidak ditemukan kandidat yang sesuai dengan filter."
    )

else:

    st.dataframe(

        candidate_display,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# 10. DISTRIBUSI JUMLAH INDIKATOR
# ============================================================

st.divider()

st.header(
    "Distribusi Jumlah Indikator Kandidat"
)

if candidates.empty:

    st.info(
        "Belum terdapat akun yang memenuhi kriteria kandidat."
    )

else:

    indicator_distribution = (

        candidates["jumlah_indikator"]
        .value_counts()
        .reindex([3, 4], fill_value=0)
        .rename_axis("Jumlah Indikator")
        .to_frame("Jumlah Akun")

    )

    st.bar_chart(
        indicator_distribution
    )


# ============================================================
# 11. DETAIL KANDIDAT
# ============================================================

st.divider()

st.header(
    "Detail Kandidat Akun"
)

if candidates.empty:

    st.info(
        "Tidak ada kandidat yang dapat ditampilkan."
    )

    st.stop()


candidate_ids = (
    candidates["user_id"]
    .astype(str)
    .tolist()
)


# Mengidentifikasi apakah tersedia kolom username

possible_profile_columns = [

    "screen_name",

    "username",

    "user_name",

    "user_screen_name",

    "name"

]

profile_column = next(

    (
        column
        for column in possible_profile_columns
        if column in df_analysis.columns
    ),

    None

)


selected_account = st.selectbox(

    "Pilih kandidat akun untuk melihat detail",

    options=candidate_ids,

    format_func=lambda user_id: get_candidate_display(

        user_id,

        profile_column,

        df_analysis.loc[
            df_analysis["user_id_str"] == user_id
        ]

    )

)


# Ambil satu baris kandidat

selected_row = candidates.loc[

    candidates["user_id"].astype(str)
    == selected_account

].iloc[0]


# Ambil semua posting milik akun tersebut

account_posts = df_analysis.loc[

    df_analysis["user_id_str"].astype(str)
    == selected_account

].copy()


# ============================================================
# 12. INFORMASI UMUM KANDIDAT
# ============================================================

st.subheader(
    "Informasi Akun"
)

st.code(
    selected_account,
    language=None
)


d1, d2, d3, d4 = st.columns(4)

d1.metric(
    "Jumlah Posting",
    f"{int(selected_row['jumlah_posting']):,}"
)

d2.metric(
    "Jumlah Pasangan",
    f"{int(selected_row['jumlah_pasangan_lintas_akun']):,}"
)

d3.metric(
    "Rata-rata Similarity",
    format_number(
        selected_row["rata_similarity"],
        4
    )
)

d4.metric(
    "Jumlah Indikator",
    f"{int(selected_row['jumlah_indikator'])}/4"
)


time_value = selected_row[
    "rata_selisih_waktu_menit"
]

st.caption(
    "Rata-rata selisih waktu: "
    + (
        f"{float(time_value):.2f} menit"
        if pd.notna(time_value)
        else "Tidak tersedia"
    )
)


# ============================================================
# 13. DETAIL EMPAT INDIKATOR
# ============================================================

st.subheader(
    "Pemeriksaan Empat Indikator"
)


indicator_detail = pd.DataFrame({

    "Indikator": [

        "Jumlah posting",

        "Pasangan lintas akun",

        "Rata-rata similarity",

        "Rata-rata selisih waktu"

    ],

    "Nilai Aktual": [

        selected_row["jumlah_posting"],

        selected_row["jumlah_pasangan_lintas_akun"],

        format_number(
            selected_row["rata_similarity"],
            4
        ),

        (
            format_number(
                selected_row["rata_selisih_waktu_menit"],
                2
            )
            if pd.notna(
                selected_row["rata_selisih_waktu_menit"]
            )
            else "Tidak tersedia"
        )

    ],

    "Kriteria": [

        f"≥ {MIN_POSTS}",

        f"≥ {MIN_CROSS_ACCOUNT_PAIRS}",

        f"≥ {MIN_ACCOUNT_SIMILARITY}",

        f"≤ {MAX_ACCOUNT_TIME_MINUTES} menit"

    ],

    "Status": [

        get_indicator_status(
            selected_row["indikator_posting"]
        ),

        get_indicator_status(
            selected_row["indikator_lintas_akun"]
        ),

        get_indicator_status(
            selected_row["indikator_similarity"]
        ),

        get_indicator_status(
            selected_row["indikator_temporal"]
        )

    ]

})


st.dataframe(

    indicator_detail,

    use_container_width=True,

    hide_index=True

)


# ============================================================
# 14. DISTRIBUSI SENTIMEN KANDIDAT
# ============================================================

st.divider()

st.subheader(
    "Analisis Sentimen Kandidat"
)


sentiment_counts = {

    "negatif": int(
        selected_row.get("jumlah_negatif", 0)
        if pd.notna(selected_row.get("jumlah_negatif", 0))
        else 0
    ),

    "netral": int(
        selected_row.get("jumlah_netral", 0)
        if pd.notna(selected_row.get("jumlah_netral", 0))
        else 0
    ),

    "positif": int(
        selected_row.get("jumlah_positif", 0)
        if pd.notna(selected_row.get("jumlah_positif", 0))
        else 0
    )

}


sentiment_total = sum(
    sentiment_counts.values()
)


sc1, sc2, sc3 = st.columns(3)

sc1.metric(
    "Negatif",
    f"{sentiment_counts['negatif']:,}"
)

sc2.metric(
    "Netral",
    f"{sentiment_counts['netral']:,}"
)

sc3.metric(
    "Positif",
    f"{sentiment_counts['positif']:,}"
)


st.write(
    "Sentimen dominan: ",
    f"**{selected_row['sentimen_dominan']}**"
)


if sentiment_total > 0:

    sentiment_chart = pd.DataFrame({

        "Sentimen": LABEL_SENTIMEN,

        "Jumlah": [
            sentiment_counts[label]
            for label in LABEL_SENTIMEN
        ]

    }).set_index("Sentimen")

    st.bar_chart(
        sentiment_chart
    )

    sentiment_percentage = pd.DataFrame({

        "Sentimen": LABEL_SENTIMEN,

        "Persentase": [

            sentiment_counts[label]
            / sentiment_total * 100

            for label in LABEL_SENTIMEN

        ]

    })

    st.dataframe(
        sentiment_percentage,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Informasi sentimen tidak tersedia untuk akun ini."
    )

st.caption(
    "Distribusi sentimen hanya digunakan sebagai informasi pendukung, "
    "bukan sebagai indikator seleksi kandidat."
)


# ============================================================
# 15. DAFTAR POSTING KANDIDAT
# ============================================================

st.divider()

st.subheader(
    "Riwayat Unggahan Akun"
)

post_columns = [

    column
    for column in [

        "id_str",

        "created_at_wib",

        "created_at",

        "full_text",

        "clean_text",

        "sentimen_analisis",

        "confidence_indobert"

    ]

    if column in account_posts.columns

]

if account_posts.empty:

    st.info(
        "Tidak ditemukan unggahan untuk akun ini."
    )

else:

    st.write(
        f"Total unggahan akun: {len(account_posts):,}"
    )

    st.dataframe(

        account_posts[post_columns],

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# 16. PASANGAN NARASI YANG MELIBATKAN KANDIDAT
# ============================================================

st.divider()

st.subheader(
    "Pasangan Narasi Akun"
)

if not df_pairs.empty:

    selected_pairs = df_pairs.loc[

        (
            df_pairs["user_id_str_1"].astype(str)
            == selected_account
        )

        |

        (
            df_pairs["user_id_str_2"].astype(str)
            == selected_account
        )

    ].copy()

else:

    selected_pairs = pd.DataFrame()


if selected_pairs.empty:

    st.info(
        "Tidak terdapat pasangan narasi terpilih "
        "yang melibatkan akun ini."
    )

else:

    pair_columns = [

        column
        for column in [

            "id_str_1",
            "id_str_2",

            "user_id_str_1",
            "user_id_str_2",

            "cosine_similarity",

            "selisih_waktu_menit",

            "sentimen_analisis_1",
            "sentimen_analisis_2",

            "keselarasan_sentimen",

            "clean_text_1",
            "clean_text_2"

        ]

        if column in selected_pairs.columns

    ]

    aligned_count = int(
        selected_pairs[
            "keselarasan_sentimen"
        ].eq("Selaras").sum()
    )

    different_count = int(
        selected_pairs[
            "keselarasan_sentimen"
        ].eq("Berbeda").sum()
    )

    pc1, pc2, pc3 = st.columns(3)

    pc1.metric(
        "Total pasangan",
        f"{len(selected_pairs):,}"
    )

    pc2.metric(
        "Sentimen selaras",
        f"{aligned_count:,}"
    )

    pc3.metric(
        "Sentimen berbeda",
        f"{different_count:,}"
    )

    st.dataframe(

        selected_pairs[pair_columns],

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# 17. HUBUNGAN DENGAN AKUN LAIN
# ============================================================

st.divider()

st.subheader(
    "Hubungan dengan Akun Lain"
)


if not df_relations.empty:

    selected_relations = df_relations.loc[

        (
            df_relations["akun_a"].astype(str)
            == selected_account
        )

        |

        (
            df_relations["akun_b"].astype(str)
            == selected_account
        )

    ].copy()

else:

    selected_relations = pd.DataFrame()


if selected_relations.empty:

    st.info(
        "Belum tersedia hubungan akun yang dapat ditampilkan."
    )

else:

    st.write(
        "Akun yang memiliki pasangan unggahan dengan kandidat ini:"
    )

    st.dataframe(

        selected_relations,

        use_container_width=True,

        hide_index=True

    )


# ============================================================
# 18. EKSPOR DAFTAR KANDIDAT DAN DETAIL
# ============================================================

st.divider()

st.header(
    "Ekspor Hasil Identifikasi Kandidat"
)


def make_candidate_excel():

    excel_buffer = io.BytesIO()

    all_candidate_ids = set(
        candidates["user_id"].astype(str)
    )

    candidate_posts = df_analysis.loc[

        df_analysis["user_id_str"].astype(str)
        .isin(all_candidate_ids)

    ].copy()


    if df_pairs.empty:

        candidate_pairs = pd.DataFrame()

    else:

        candidate_pairs = df_pairs.loc[

            df_pairs["user_id_str_1"].astype(str)
            .isin(all_candidate_ids)

            |

            df_pairs["user_id_str_2"].astype(str)
            .isin(all_candidate_ids)

        ].copy()


    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        excel_safe_dataframe(
            candidates
        ).to_excel(

            writer,

            index=False,

            sheet_name="Daftar_Kandidat"

        )

        excel_safe_dataframe(
            indicators
        ).to_excel(

            writer,

            index=False,

            sheet_name="Seluruh_Indikator"

        )

        excel_safe_dataframe(
            candidate_posts
        ).to_excel(

            writer,

            index=False,

            sheet_name="Posting_Kandidat"

        )

        excel_safe_dataframe(
            candidate_pairs
        ).to_excel(

            writer,

            index=False,

            sheet_name="Pasangan_Narasi"

        )

        excel_safe_dataframe(
            df_relations
        ).to_excel(

            writer,

            index=False,

            sheet_name="Hubungan_Akun"

        )

    excel_buffer.seek(0)

    return excel_buffer.getvalue()


download_col1, download_col2 = st.columns(2)


with download_col1:

    candidate_csv = candidates.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(

        label="Download Daftar Kandidat (CSV)",

        data=candidate_csv,

        file_name="daftar_kandidat_akun.csv",

        mime="text/csv",

        use_container_width=True

    )


with download_col2:

    st.download_button(

        label="Download Seluruh Detail (Excel)",

        data=make_candidate_excel(),

        file_name="hasil_identifikasi_kandidat_akun.xlsx",

        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),

        type="primary",

        use_container_width=True

    )


# ============================================================
# 19. CATATAN METODOLOGIS
# ============================================================

st.divider()

st.info(
    """
    **Catatan interpretasi**

    Akun yang ditampilkan merupakan kandidat akun dengan indikasi
    aktivitas komunikasi terkoordinasi berdasarkan aturan penelitian.

    Hasil ini tidak membuktikan bahwa akun tersebut merupakan buzzer
    atau bahwa seluruh aktivitasnya dilakukan secara terkoordinasi.

    Keseragaman sentimen digunakan sebagai informasi pendukung,
    sedangkan seleksi kandidat tetap didasarkan pada empat indikator
    aktivitas akun.
    """
)


# ============================================================
# 20. FOOTER
# ============================================================

st.divider()

st.caption(
    "Buzzer Analytics | Identifikasi Kandidat Akun"
)
