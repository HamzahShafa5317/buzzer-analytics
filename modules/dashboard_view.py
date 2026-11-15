
import pandas as pd
import streamlit as st


# ============================================================
# 1. FUNGSI PENDUKUNG
# ============================================================

def get_dataframe(data, key):

    value = data.get(key)

    if isinstance(value, pd.DataFrame):
        return value.copy()

    return pd.DataFrame()


def format_number(value):

    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def get_sentiment_counts(df, column):

    labels = [
        "negatif",
        "netral",
        "positif"
    ]

    if df.empty or column not in df.columns:

        return pd.Series(
            0,
            index=labels,
            dtype=int
        )

    values = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    return (
        values
        .value_counts()
        .reindex(labels, fill_value=0)
        .astype(int)
    )


# ============================================================
# 2. FUNGSI UTAMA DASHBOARD
# ============================================================

def render_dashboard():

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title("BUZZER ANALYTICS")

    st.caption(
        "Social Media Coordination Analysis Platform"
    )

    st.write(
        """
        Platform analisis media sosial yang mengintegrasikan
        klasifikasi sentimen, kemiripan narasi, pola temporal,
        dan hubungan antar akun untuk mengidentifikasi
        indikasi aktivitas komunikasi terkoordinasi.
        """
    )

    st.divider()


    # --------------------------------------------------------
    # AMBIL DATA SESSION STATE
    # --------------------------------------------------------

    uploaded_df = st.session_state.get(
        "uploaded_df",
        pd.DataFrame()
    )

    preprocessed_df = st.session_state.get(
        "preprocessed_df",
        pd.DataFrame()
    )

    sentiment_df = st.session_state.get(
        "sentiment_df",
        pd.DataFrame()
    )

    narrative_results = st.session_state.get(
        "narrative_results",
        {}
    )

    if not isinstance(narrative_results, dict):
        narrative_results = {}

    df_analysis = get_dataframe(
        narrative_results,
        "df_analysis"
    )

    df_pairs = get_dataframe(
        narrative_results,
        "df_sentimen_pasangan"
    )

    df_candidates = get_dataframe(
        narrative_results,
        "df_kandidat_final"
    )

    df_relations = get_dataframe(
        narrative_results,
        "df_hubungan_akun"
    )


    # --------------------------------------------------------
    # STATUS PROSES ANALISIS
    # --------------------------------------------------------

    has_upload = (
        isinstance(uploaded_df, pd.DataFrame)
        and not uploaded_df.empty
    )

    has_preprocessing = (
        isinstance(preprocessed_df, pd.DataFrame)
        and not preprocessed_df.empty
    )

    has_sentiment = (
        isinstance(sentiment_df, pd.DataFrame)
        and "sentimen_indobert" in sentiment_df.columns
    )

    has_narrative = (
        not df_analysis.empty
        and "sentimen_analisis" in df_analysis.columns
    )


    # --------------------------------------------------------
    # INFORMASI STATUS
    # --------------------------------------------------------

    st.subheader("Status Analisis")

    status_cols = st.columns(4)

    status_items = [
        (
            "Upload Dataset",
            has_upload
        ),
        (
            "Preprocessing",
            has_preprocessing
        ),
        (
            "IndoBERT",
            has_sentiment
        ),
        (
            "Sentence-BERT",
            has_narrative
        )
    ]

    for col, (name, completed) in zip(
        status_cols,
        status_items
    ):

        with col:

            if completed:
                st.success(name + " selesai")
            else:
                st.warning(name + " belum tersedia")

    st.divider()


    # --------------------------------------------------------
    # JUMLAH DATASET
    # --------------------------------------------------------

    total_upload = (
        len(uploaded_df)
        if has_upload
        else 0
    )

    total_preprocessed = (
        len(preprocessed_df)
        if has_preprocessing
        else 0
    )

    total_accounts = 0

    if has_narrative and "user_id_str" in df_analysis.columns:

        total_accounts = (
            df_analysis["user_id_str"]
            .astype("string")
            .nunique()
        )


    # --------------------------------------------------------
    # HASIL SENTIMEN
    # --------------------------------------------------------

    sentiment_counts = get_sentiment_counts(
        sentiment_df,
        "sentimen_indobert"
    )

    total_predictions = int(
        sentiment_counts.sum()
    )

    # Jika hasil prediksi belum ada,
    # tampilkan nilai nol.
    if not has_sentiment:

        sentiment_counts[:] = 0
        total_predictions = 0


    # --------------------------------------------------------
    # HASIL NARASI
    # --------------------------------------------------------

    total_analysis = len(df_analysis)

    total_pairs = len(df_pairs)

    total_candidates = len(df_candidates)

    total_relations = len(df_relations)


    candidate_percentage = (

        total_candidates / total_accounts * 100

        if total_accounts > 0

        else 0

    )


    # ========================================================
    # 3. KPI UTAMA
    # ========================================================

    st.header("Ringkasan Hasil Analisis")

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Total Unggahan",
        format_number(total_upload)
    )

    k2.metric(
        "Akun Dianalisis",
        format_number(total_accounts)
    )

    k3.metric(
        "Pasangan Narasi Terpilih",
        format_number(total_pairs)
    )

    k4.metric(
        "Kandidat Akun",
        format_number(total_candidates)
    )

    if not has_upload:

        st.info(
            "Belum ada dataset. "
            "Silakan buka halaman upload pada Dashboard "
            "untuk memulai analisis."
        )

    elif not has_sentiment:

        st.info(
            "Dataset telah dimuat, tetapi analisis IndoBERT "
            "belum dijalankan."
        )

    elif not has_narrative:

        st.info(
            "Analisis sentimen tersedia. "
            "Jalankan Sentence-BERT untuk melengkapi dashboard."
        )

    st.divider()


    # ========================================================
    # 4. DISTRIBUSI SENTIMEN
    # ========================================================

    st.header("Analisis Sentimen")

    if has_sentiment:

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Prediksi",
            format_number(total_predictions)
        )

        c2.metric(
            "Negatif",
            format_number(sentiment_counts["negatif"])
        )

        c3.metric(
            "Netral",
            format_number(sentiment_counts["netral"])
        )

        c4.metric(
            "Positif",
            format_number(sentiment_counts["positif"])
        )

        sentiment_chart = pd.DataFrame({

            "Jumlah": sentiment_counts

        })

        st.bar_chart(
            sentiment_chart,
            width="stretch"
        )

        if total_predictions > 0:

            percentage_df = pd.DataFrame({

                "Sentimen": sentiment_counts.index,

                "Jumlah": sentiment_counts.values,

                "Persentase": (

                    sentiment_counts.values
                    / total_predictions
                    * 100

                )

            })

            st.dataframe(
                percentage_df,
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "Distribusi sentimen akan ditampilkan "
            "setelah analisis IndoBERT selesai."
        )

    st.divider()


    # ========================================================
    # 5. RINGKASAN ANALISIS NARASI
    # ========================================================

    st.header("Analisis Kemiripan Narasi")

    if has_narrative:

        n1, n2, n3, n4 = st.columns(4)

        n1.metric(
            "Data Analisis",
            format_number(total_analysis)
        )

        n2.metric(
            "Akun Dianalisis",
            format_number(total_accounts)
        )

        n3.metric(
            "Pasangan Narasi",
            format_number(total_pairs)
        )

        n4.metric(
            "Hubungan Akun",
            format_number(total_relations)
        )


        # ----------------------------------------------------
        # STATISTIK SIMILARITY
        # ----------------------------------------------------

        if (
            not df_pairs.empty
            and "cosine_similarity" in df_pairs.columns
        ):

            similarities = pd.to_numeric(
                df_pairs["cosine_similarity"],
                errors="coerce"
            ).dropna()

            if not similarities.empty:

                sim1, sim2, sim3 = st.columns(3)

                sim1.metric(
                    "Similarity Rata-rata",
                    f"{similarities.mean():.4f}"
                )

                sim2.metric(
                    "Similarity Tertinggi",
                    f"{similarities.max():.4f}"
                )

                sim3.metric(
                    "Similarity Terendah",
                    f"{similarities.min():.4f}"
                )


        # ----------------------------------------------------
        # KESELARASAN SENTIMEN PASANGAN
        # ----------------------------------------------------

        if (
            not df_pairs.empty
            and "keselarasan_sentimen" in df_pairs.columns
        ):

            alignment = (
                df_pairs["keselarasan_sentimen"]
                .astype("string")
                .str.strip()
                .str.casefold()
            )

            aligned = int(
                alignment.eq("selaras").sum()
            )

            different = int(
                alignment.eq("berbeda").sum()
            )

            a1, a2 = st.columns(2)

            a1.metric(
                "Pasangan Sentimen Selaras",
                format_number(aligned)
            )

            a2.metric(
                "Pasangan Sentimen Berbeda",
                format_number(different)
            )

    else:

        st.info(
            "Hasil kemiripan narasi akan ditampilkan "
            "setelah Sentence-BERT selesai dijalankan."
        )

    st.divider()


    # ========================================================
    # 6. IDENTIFIKASI KANDIDAT AKUN
    # ========================================================

    st.header("Identifikasi Kandidat Akun")

    if has_narrative:

        if df_candidates.empty:

            st.info(
                "Belum terdapat akun yang memenuhi "
                "kriteria kandidat penelitian."
            )

        else:

            candidate_counts = (
                df_candidates["jumlah_indikator"]
                .value_counts()
                .reindex([3, 4], fill_value=0)
            )

            candidate_cols = st.columns(3)

            candidate_cols[0].metric(
                "Total Kandidat",
                format_number(total_candidates)
            )

            candidate_cols[1].metric(
                "Memenuhi 3 Indikator",
                format_number(candidate_counts[3])
            )

            candidate_cols[2].metric(
                "Memenuhi 4 Indikator",
                format_number(candidate_counts[4])
            )


            # ------------------------------------------------
            # SENTIMEN DOMINAN KANDIDAT
            # ------------------------------------------------

            if "sentimen_dominan" in df_candidates.columns:

                candidate_sentiment = (
                    df_candidates["sentimen_dominan"]
                    .astype("string")
                    .str.lower()
                    .value_counts()
                    .reindex(
                        [
                            "negatif",
                            "netral",
                            "positif"
                        ],
                        fill_value=0
                    )
                )

                st.subheader(
                    "Sentimen Dominan Kandidat"
                )

                st.bar_chart(
                    candidate_sentiment.to_frame(
                        name="Jumlah Akun"
                    ),
                    width="stretch"
                )


            # ------------------------------------------------
            # DAFTAR KANDIDAT TERATAS
            # ------------------------------------------------

            st.subheader(
                "Ringkasan Kandidat Akun"
            )

            display_columns = [

                column
                for column in [

                    "user_id",

                    "jumlah_posting",

                    "jumlah_pasangan_lintas_akun",

                    "rata_similarity",

                    "rata_selisih_waktu_menit",

                    "jumlah_indikator",

                    "sentimen_dominan"

                ]

                if column in df_candidates.columns

            ]

            if display_columns:

                top_candidates = (
                    df_candidates
                    .sort_values(
                        by=[
                            "jumlah_indikator",
                            "rata_similarity"
                        ],
                        ascending=[
                            False,
                            False
                        ]
                    )
                    .head(10)
                )

                st.dataframe(
                    top_candidates[display_columns],
                    width="stretch",
                    hide_index=True
                )

    else:

        st.info(
            "Identifikasi kandidat akan muncul "
            "setelah analisis narasi selesai."
        )

    st.divider()


    # ========================================================
    # 7. HUBUNGAN ANTAR AKUN
    # ========================================================

    st.header("Hubungan Antar Akun")

    if has_narrative and not df_relations.empty:

        relation_columns = [

            column
            for column in [

                "akun_a",

                "akun_b",

                "jumlah_pasangan",

                "rata_similarity",

                "maks_similarity",

                "rata_selisih_waktu_menit"

            ]

            if column in df_relations.columns

        ]

        st.write(
            "Hubungan akun yang terbentuk dari pasangan "
            "unggahan dengan narasi serupa."
        )

        st.dataframe(
            df_relations[relation_columns].head(15),
            width="stretch",
            hide_index=True
        )

    elif has_narrative:

        st.info(
            "Belum ditemukan hubungan akun untuk ditampilkan."
        )

    else:

        st.info(
            "Hubungan akun akan muncul setelah Sentence-BERT selesai."
        )

    st.divider()


    # ========================================================
    # 8. CATATAN DAN INTERPRETASI
    # ========================================================

    st.header("Interpretasi Hasil Analisis")

    st.write(
        """
        Sistem mengintegrasikan hasil klasifikasi sentimen
        menggunakan IndoBERT dengan pengukuran kemiripan
        semantik menggunakan Sentence-BERT.

        Identifikasi kandidat akun dilakukan berdasarkan
        empat indikator, yaitu frekuensi posting, pasangan
        lintas akun, rata-rata similarity, dan pola temporal.
        """
    )

    st.info(
        """
        Akun yang teridentifikasi merupakan kandidat dengan
        indikasi aktivitas komunikasi terkoordinasi berdasarkan
        kriteria penelitian. Hasil analisis tidak membuktikan
        identitas buzzer atau memastikan adanya koordinasi
        secara sengaja. Sentimen digunakan sebagai informasi
        pendukung, bukan sebagai indikator seleksi kandidat.
        """
    )


    # ========================================================
    # 9. INFORMASI DATASET DAN VALIDITAS
    # ========================================================

    st.header("Informasi Dataset")

    if has_upload:

        info_data = [

            {
                "Informasi": "Nama file",
                "Nilai": st.session_state.get(
                    "uploaded_filename",
                    "-"
                )
            },

            {
                "Informasi": "Jumlah data awal",
                "Nilai": format_number(total_upload)
            },

            {
                "Informasi": "Jumlah data preprocessing",
                "Nilai": format_number(total_preprocessed)
            },

            {
                "Informasi": "Jumlah data prediksi",
                "Nilai": format_number(total_predictions)
            },

            {
                "Informasi": "Jumlah data analisis narasi",
                "Nilai": format_number(total_analysis)
            },

            {
                "Informasi": "Jumlah akun dianalisis",
                "Nilai": format_number(total_accounts)
            }

        ]

        st.dataframe(
            pd.DataFrame(info_data),
            width="stretch",
            hide_index=True
        )


    # --------------------------------------------------------
    # PERINGATAN RELEVANSI
    # --------------------------------------------------------

    relevance_available = (
        has_preprocessing
        and "relevansi" in preprocessed_df.columns
    )

    if has_narrative and not relevance_available:

        st.warning(
            """
            Dataset tidak memiliki label relevansi. Oleh karena itu,
            semua unggahan dengan teks tidak kosong dapat masuk ke
            analisis narasi. Jumlah data dan hasil pasangan dapat
            berbeda dari pipeline penelitian yang melakukan
            filtering unggahan tidak relevan.
            """
        )


    # ========================================================
    # 10. FOOTER
    # ========================================================

    st.divider()

    st.caption(
        "BUZZER ANALYTICS | "
        "Integrasi Analisis Sentimen dan Analisis Narasi"
    )
