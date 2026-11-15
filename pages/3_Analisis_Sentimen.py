
import streamlit as st
import pandas as pd

from io import BytesIO

from modules.sentiment_analysis import (
    load_indobert_model,
    predict_sentiment_dataframe
)

from config import (
    INFERENCE_MAX_LENGTH,
    INFERENCE_BATCH_SIZE,
    ID2LABEL
)


# ============================================================
# 1. CACHE MODEL
# Model cukup dimuat satu kali selama sesi aplikasi.
# ============================================================

@st.cache_resource(show_spinner=False)
def get_indobert_bundle():

    return load_indobert_model()


# ============================================================
# 2. HEADER
# ============================================================

st.title("Analisis Sentimen IndoBERT")

st.caption(
    "Klasifikasi sentimen unggahan media sosial "
    "menggunakan IndoBERT Eksperimen F."
)

st.divider()


# ============================================================
# 3. VALIDASI DATA PREPROCESSING
# ============================================================

if "preprocessed_df" not in st.session_state:

    st.warning(
        "Dataset belum melalui preprocessing. "
        "Silakan kembali ke halaman utama."
    )

    st.stop()


df = st.session_state[
    "preprocessed_df"
].copy()


if "clean_text" not in df.columns:

    st.error(
        "Kolom clean_text tidak ditemukan."
    )

    st.stop()


# ============================================================
# 4. INFORMASI DATASET
# ============================================================

st.header("Dataset Siap Analisis")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Unggahan",
    f"{len(df):,}"
)

col2.metric(
    "Teks Tidak Kosong",
    f"{df['clean_text'].fillna('').astype(str).str.strip().ne('').sum():,}"
)

col3.metric(
    "Panjang Token",
    str(INFERENCE_MAX_LENGTH)
)


# ============================================================
# 5. INFORMASI MODEL
# ============================================================

st.divider()

st.subheader("Model yang Digunakan")

info1, info2, info3 = st.columns(3)

info1.metric(
    "Model",
    "IndoBERT"
)

info2.metric(
    "Checkpoint",
    "470"
)

info3.metric(
    "Batch Size",
    str(INFERENCE_BATCH_SIZE)
)

st.caption(
    "Model menggunakan tiga kelas: Negatif, Netral, dan Positif."
)


# ============================================================
# 6. TOMBOL ANALISIS SENTIMEN
# ============================================================

st.divider()

st.header("Jalankan Analisis Sentimen")

st.write(
    """
    Sistem akan memproses setiap teks yang tidak kosong
    menggunakan model IndoBERT yang telah melalui fine-tuning.
    """
)


if st.button(
    "Jalankan Analisis Sentimen",
    type="primary",
    use_container_width=True,
    key="btn_sentiment"
):

    try:

        with st.spinner(
            "Memuat model IndoBERT..."
        ):

            tokenizer, model, device = (
                get_indobert_bundle()
            )

        st.info(
            f"Perangkat komputasi: {device}"
        )

        progress_bar = st.progress(0)

        progress_text = st.empty()

        def update_progress(done, total):

            percentage = (
                done / total
                if total > 0
                else 1.0
            )

            progress_bar.progress(
                min(percentage, 1.0)
            )

            progress_text.text(
                f"Memproses {done:,} dari {total:,} unggahan"
            )

        # Jalankan prediksi
        with st.spinner(
            "Sedang melakukan klasifikasi sentimen..."
        ):

            result = predict_sentiment_dataframe(
                df=df,
                tokenizer=tokenizer,
                model=model,
                device=device,
                batch_size=INFERENCE_BATCH_SIZE,
                max_length=INFERENCE_MAX_LENGTH,
                progress_callback=update_progress
            )

        # Simpan hasil
        st.session_state[
            "sentiment_df"
        ] = result
        
        # Hapus hasil integrasi lama jika prediksi dijalankan ulang
        for key in [
            "integrated_df",
            "analysis_df",
            "integration_summary"
        ]:
            st.session_state.pop(key, None)
        

        # Statistik
        distribution = (
            result["sentimen_indobert"]
            .value_counts()
            .to_dict()
        )

        st.session_state[
            "sentiment_summary"
        ] = distribution

        progress_bar.progress(1.0)

        progress_text.success(
            "Analisis sentimen selesai."
        )

        st.success(
            "Prediksi IndoBERT berhasil diselesaikan!"
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan pada analisis sentimen: {e}"
        )


# ============================================================
# 7. MENAMPILKAN HASIL SENTIMEN
# ============================================================

if "sentiment_df" in st.session_state:

    result = st.session_state[
        "sentiment_df"
    ]

    st.divider()

    st.header("Hasil Analisis Sentimen")


    # --------------------------------------------------------
    # DISTRIBUSI SENTIMEN
    # --------------------------------------------------------

    counts = (
        result["sentimen_indobert"]
        .value_counts()
        .reindex(
            ["negatif", "netral", "positif"],
            fill_value=0
        )
    )

    total_predicted = int(
        counts.sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Prediksi",
        f"{total_predicted:,}"
    )

    c2.metric(
        "Negatif",
        f"{counts['negatif']:,}"
    )

    c3.metric(
        "Netral",
        f"{counts['netral']:,}"
    )

    c4.metric(
        "Positif",
        f"{counts['positif']:,}"
    )


    # --------------------------------------------------------
    # GRAFIK DISTRIBUSI
    # --------------------------------------------------------

    st.subheader(
        "Distribusi Kelas Sentimen"
    )

    distribution_df = pd.DataFrame({

        "Sentimen": counts.index,

        "Jumlah": counts.values

    }).set_index("Sentimen")

    st.bar_chart(
        distribution_df
    )


    # --------------------------------------------------------
    # RATA-RATA CONFIDENCE
    # --------------------------------------------------------

    mean_confidence = (
        result["confidence_indobert"]
        .mean()
    )

    if pd.notna(mean_confidence):

        st.metric(
            "Rata-rata Confidence",
            f"{mean_confidence:.2%}"
        )


    # --------------------------------------------------------
    # PREVIEW PREDIKSI
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Preview Hasil Prediksi"
    )

    preview_columns = [

        column for column in [

            "id_str",
            "user_id_str",
            "full_text",
            "clean_text",
            "sentimen_indobert",
            "confidence_indobert",
            "status_analisis_sentimen"

        ]

        if column in result.columns

    ]

    st.dataframe(
        result[preview_columns].head(50),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # FILTER SENTIMEN
    # --------------------------------------------------------

    st.subheader(
        "Filter Hasil Prediksi"
    )

    available_labels = [
        label
        for label in [
            "negatif",
            "netral",
            "positif"
        ]
        if counts[label] > 0
    ]

    selected_labels = st.multiselect(
        "Pilih sentimen",
        options=available_labels,
        default=available_labels
    )

    filtered = result[
        result["sentimen_indobert"].isin(
            selected_labels
        )
    ]

    st.write(
        f"Jumlah data yang ditampilkan: {len(filtered):,}"
    )

    st.dataframe(
        filtered[preview_columns].head(500),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # EKSPOR HASIL
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Ekspor Hasil Analisis Sentimen"
    )

    csv_data = result.to_csv(
        index=False
    ).encode("utf-8-sig")

    excel_buffer = BytesIO()

    with pd.ExcelWriter(
        excel_buffer,
        engine="openpyxl"
    ) as writer:

        result.to_excel(
            writer,
            index=False,
            sheet_name="Hasil_Sentimen"
        )

        counts.rename(
            "Jumlah"
        ).to_frame().to_excel(
            writer,
            sheet_name="Distribusi"
        )

    download1, download2 = st.columns(2)

    with download1:

        st.download_button(
            "Download CSV",
            data=csv_data,
            file_name="hasil_sentimen_indobert.xlsx.csv",
            mime="text/csv",
            use_container_width=True
        )

    with download2:

        st.download_button(
            "Download Excel",
            data=excel_buffer.getvalue(),
            file_name="hasil_sentimen_indobert.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
            use_container_width=True
        )


else:

    st.info(
        "Belum ada hasil prediksi. "
        "Klik tombol Jalankan Analisis Sentimen."
    )
