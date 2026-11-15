
import streamlit as st
import pandas as pd

from modules.dashboard_view import render_dashboard
from io import BytesIO

from modules.data_loader import (
    standardize_columns,
    validate_dataset,
    get_dataset_summary
)

from modules.preprocessing import (
    run_preprocessing
)


# ============================================================
# 1. KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Analisis Indikasi Aktivitas Buzzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. FUNGSI MEMBACA DATASET
# ============================================================

def read_dataset(uploaded_file, sheet_name=None):

    file_extension = (
        uploaded_file.name.split(".")[-1].lower()
    )

    uploaded_file.seek(0)

    if file_extension in ["xlsx", "xls"]:

        df = pd.read_excel(
            uploaded_file,
            sheet_name=sheet_name,
            dtype=str,
            keep_default_na=False
        )

    elif file_extension == "csv":

        df = pd.read_csv(
            uploaded_file,
            dtype=str,
            keep_default_na=False,
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

    else:

        raise ValueError(
            "Format file tidak didukung."
        )

    return df


# ============================================================
# 3. FUNGSI RESET HASIL ANALISIS
# ============================================================


def reset_analysis():

    keys_to_remove = [

        "preprocessed_df",

        "preprocessing_summary",

        "sentiment_df",

        "sentiment_summary",

        "integrated_df",

        "analysis_df",

        "integration_summary",

        "annotation_raw"

    ]

    for key in keys_to_remove:

        st.session_state.pop(
            key,
            None
        )


# ============================================================
# 4. HEADER WEBSITE
# ============================================================

st.title(
    "Analisis Indikasi Aktivitas Buzzer"
)

st.caption(
    "Integrasi Analisis Sentimen IndoBERT "
    "dan Analisis Narasi Sentence-BERT"
)

st.divider()


# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("BUZZER ANALYTICS")

    st.caption(
        "Sistem Analisis Media Sosial"
    )

    st.divider()

    st.info(
        "Tahap 1: Upload dan Validasi Dataset"
    )

    st.info(
        "Tahap 2: Preprocessing Dataset"
    )

    st.caption(
        "Pengembangan sistem untuk penelitian skripsi."
    )

    # ============================================================
# DASHBOARD UTAMA
# ============================================================

render_dashboard()

st.divider()


# ============================================================
# 6. UPLOAD DATASET
# ============================================================

st.header("Tahap 1: Upload Dataset")

st.write(
    """
    Masukkan dataset unggahan media sosial X.
    

    Format file yang didukung adalah CSV dan Excel.
    """
)

uploaded_file = st.file_uploader(
    "Pilih file dataset",
    type=["csv", "xlsx", "xls"],
    key="dataset_uploader"
)


# ============================================================
# 7. PEMILIHAN SHEET EXCEL
# ============================================================

selected_sheet = None

if uploaded_file is not None:

    file_extension = (
        uploaded_file.name.split(".")[-1].lower()
    )

    if file_extension in ["xlsx", "xls"]:

        try:

            uploaded_file.seek(0)

            excel_file = pd.ExcelFile(
                uploaded_file
            )

            selected_sheet = st.selectbox(
                "Pilih sheet dataset",
                excel_file.sheet_names
            )

        except Exception as e:

            st.error(
                f"Gagal membaca sheet Excel: {e}"
            )


# ============================================================
# 8. TOMBOL MEMUAT DATASET
# ============================================================

if uploaded_file is not None:

    if st.button(
        "Muat dan Periksa Dataset",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Sedang membaca dataset..."
            ):

                df = read_dataset(
                    uploaded_file,
                    selected_sheet
                )

                df = standardize_columns(df)

                missing_columns = validate_dataset(
                    df
                )

            if missing_columns:

                st.error(
                    "Dataset belum memenuhi struktur yang diperlukan."
                )

                st.write(
                    "Kolom yang belum ditemukan:",
                    missing_columns
                )

            else:

                if len(df) == 0:

                    st.error(
                        "Dataset tidak memiliki baris data."
                    )

                    st.stop()

                # Simpan dataset
                st.session_state[
                    "uploaded_df"
                ] = df

                st.session_state[
                    "uploaded_filename"
                ] = uploaded_file.name

                # Hapus hasil analisis sebelumnya
                reset_analysis()

                st.success(
                    "Dataset berhasil dimuat!"
                )

        except Exception as e:

            st.error(
                f"Gagal memuat dataset: {e}"
            )


# ============================================================
# 9. MENAMPILKAN DATASET YANG SUDAH DIMUAT
# ============================================================

if "uploaded_df" in st.session_state:

    df = st.session_state[
        "uploaded_df"
    ]

    upload_summary = get_dataset_summary(
        df
    )

    st.divider()

    st.header("Ringkasan Dataset")


    # --------------------------------------------------------
    # STATISTIK DATASET
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Unggahan",
        f"{upload_summary['total_data']:,}"
    )

    col2.metric(
        "Akun Unik",
        f"{upload_summary['jumlah_akun']:,}"
    )

    col3.metric(
        "Teks Kosong",
        f"{upload_summary['teks_kosong']:,}"
    )

    col4.metric(
        "ID Duplikat",
        f"{upload_summary['id_duplikat']:,}"
    )


    # --------------------------------------------------------
    # PREVIEW DATASET
    # --------------------------------------------------------

    st.divider()

    st.subheader("Preview Dataset")

    display_columns = [
        column
        for column in [
            "id_str",
            "user_id_str",
            "created_at",
            "full_text",
            "clean_text"
        ]
        if column in df.columns
    ]

    st.dataframe(
        df[display_columns].head(20),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        f"File: {st.session_state['uploaded_filename']} "
        f"| Total baris: {len(df):,}"
    )


    # ========================================================
    # 10. TAHAP 2: PREPROCESSING DATASET
    # ========================================================

    st.divider()

    st.header("Tahap 2: Preprocessing Dataset")

    st.write(
        """
        Proses preprocessing dilakukan untuk membersihkan
        teks unggahan sebelum digunakan dalam analisis sentimen
        dan analisis narasi.
        """
    )


    # --------------------------------------------------------
    # PENJELASAN TAHAPAN PREPROCESSING
    # --------------------------------------------------------

    with st.expander(
        "Lihat tahapan preprocessing",
        expanded=False
    ):

        preprocessing_steps = [
            "Case Folding",
            "Cleaning Text",
            "Normalisasi Repeated Character",
            "Remove URL",
            "Remove Mention",
            "Hashtag Processing",
            "Remove Emoji",
            "Remove Duplicate berdasarkan ID posting"
        ]

        for index, step in enumerate(
            preprocessing_steps,
            start=1
        ):

            st.write(
                f"{index}. {step}"
            )


    # --------------------------------------------------------
    # TOMBOL PREPROCESSING
    # --------------------------------------------------------

    if st.button(
        "Jalankan Preprocessing",
        type="primary",
        use_container_width=True,
        key="btn_preprocessing"
    ):

        try:

            with st.spinner(
                "Sedang melakukan preprocessing dataset..."
            ):

                df_processed, prep_summary = (
                    run_preprocessing(
                        df.copy()
                    )
                )

                st.session_state[
                    "preprocessed_df"
                ] = df_processed

                st.session_state[
                    "preprocessing_summary"
                ] = prep_summary

            st.success(
                "Preprocessing berhasil diselesaikan!"
            )

        except Exception as e:

            st.error(
                f"Terjadi kesalahan preprocessing: {e}"
            )


    # ========================================================
    # 11. MENAMPILKAN HASIL PREPROCESSING
    # ========================================================

    if "preprocessed_df" in st.session_state:

        df_processed = st.session_state[
            "preprocessed_df"
        ]

        prep_summary = st.session_state[
            "preprocessing_summary"
        ]

        validation = prep_summary[
            "validasi"
        ]

        st.divider()

        st.header("Hasil Preprocessing")


        # ----------------------------------------------------
        # STATISTIK HASIL
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Data Awal",
            f"{prep_summary['jumlah_data_awal']:,}"
        )

        c2.metric(
            "ID Duplikat Dihapus",
            f"{prep_summary['jumlah_data_duplikat']:,}"
        )

        c3.metric(
            "Data Akhir",
            f"{prep_summary['jumlah_data_akhir']:,}"
        )

        c4.metric(
            "Akun Unik",
            f"{prep_summary['jumlah_akun_unik']:,}"
        )


        # ----------------------------------------------------
        # TABEL VALIDASI
        # ----------------------------------------------------

        st.subheader(
            "Pemeriksaan Hasil Preprocessing"
        )

        validation_descriptions = {

            "total_data":
                "Jumlah data setelah deduplikasi",

            "teks_kosong":
                "Teks kosong",

            "sisa_url":
                "URL yang masih ditemukan",

            "sisa_mention":
                "Mention yang masih ditemukan",

            "sisa_hashtag":
                "Simbol hashtag yang masih ditemukan",

            "sisa_repeated_character":
                "Karakter berulang yang masih ditemukan",

            "sisa_punctuation":
                "Tanda baca berulang yang masih ditemukan"

        }

        validation_rows = []

        for key, value in validation.items():

            if key == "total_data":

                status = "Informasi"

            elif value > 0:

                status = "Perlu diperiksa"

            else:

                status = "OK"

            validation_rows.append({

                "Pemeriksaan": (
                    validation_descriptions.get(
                        key,
                        key
                    )
                ),

                "Jumlah": value,

                "Status": status

            })

        validation_df = pd.DataFrame(
            validation_rows
        )

        st.dataframe(
            validation_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # PREVIEW CLEAN TEXT
        # ----------------------------------------------------

        st.subheader(
            "Preview Hasil Clean Text"
        )

        preview_columns = [
            column
            for column in [
                "id_str",
                "user_id_str",
                "full_text",
                "clean_text"
            ]
            if column in df_processed.columns
        ]

        st.dataframe(
            df_processed[
                preview_columns
            ].head(20),

            use_container_width=True,

            hide_index=True
        )


        # ----------------------------------------------------
        # PERINGATAN TEKS KOSONG
        # ----------------------------------------------------

        if validation["teks_kosong"] > 0:

            st.warning(
                f"Ditemukan {validation['teks_kosong']} "
                "teks kosong setelah preprocessing. "
                "Data perlu diperiksa pada tahap analisis berikutnya."
            )

        else:

            st.success(
                "Tidak ditemukan teks kosong setelah preprocessing."
            )


        # ----------------------------------------------------
        # 12. EKSPOR HASIL CSV DAN EXCEL
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "Ekspor Dataset Hasil Preprocessing"
        )

        csv_data = df_processed.to_csv(
            index=False
        ).encode("utf-8-sig")

        excel_buffer = BytesIO()

        try:

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                df_processed.to_excel(
                    writer,
                    index=False,
                    sheet_name="Preprocessed"
                )

                validation_df.to_excel(
                    writer,
                    index=False,
                    sheet_name="Validasi"
                )

            download_col1, download_col2 = (
                st.columns(2)
            )

            with download_col1:

                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="dataset_mbg_preprocessed.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with download_col2:

                st.download_button(
                    label="Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name="dataset_mbg_preprocessed.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"Gagal membuat file ekspor: {e}"
            )


else:

    st.info(
        "Belum ada dataset yang dimuat. "
        "Silakan upload file terlebih dahulu."
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.divider()

st.caption(
    "Buzzer Analytics | Sistem Analisis Indikasi Aktivitas Buzzer"
)
