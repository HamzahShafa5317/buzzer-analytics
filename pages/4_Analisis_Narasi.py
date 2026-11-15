import io
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer

from config import SENTENCE_BERT_MODEL, SENTENCE_BERT_DIR
from modules.narrative_analysis import run_narrative_analysis


# ============================================================
# 1. KONFIGURASI DAN PEMUATAN MODEL
# ============================================================

st.set_page_config(
    page_title="Analisis Narasi Sentence-BERT",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_sentence_bert_model():
    """Memuat model lokal jika tersedia; jika belum, unduh dari Hugging Face."""
    local_dir = Path(SENTENCE_BERT_DIR)

    if (local_dir / "modules.json").exists():
        model_source = str(local_dir)
    else:
        model_source = SENTENCE_BERT_MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_source, device=device)
    return model, device, model_source


def excel_safe_dataframe(frame):
    """Mengubah datetime timezone-aware agar dapat disimpan ke Excel."""
    output = frame.copy()
    for column in output.columns:
        if isinstance(output[column].dtype, pd.DatetimeTZDtype):
            output[column] = output[column].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return output


def make_excel_export(results):
    sheets = {
        "Ringkasan": pd.DataFrame(
            [{"Komponen": key, "Nilai": value}
             for key, value in results["summary"].items()]
        ),
        "Data_Analisis": results["df_analysis"],
        "Pairs_Initial_080": results["df_pairs_initial"],
        "Base_Cross_Account": results["df_base_cross"],
        "High_Similarity_Cross": results["df_high_semantic_cross"],
        "Pairs_Selected": results["df_pairs_selected"],
        "Account_Indicators": results["df_indikator"],
        "Candidate_Accounts": results["df_kandidat_final"],
        "Candidate_Sentiment": results["df_ringkasan_sentimen"],
        "Pair_Sentiment": results["df_sentimen_pasangan"],
        "Account_Relations": results["df_hubungan_akun"],
        "Account_Label_Map": results["df_pemetaan_akun"],
        "Time_Gap_Distribution": results["df_distribusi_temporal"],
        "Sensitivity": results["df_sensitivitas"],
    }

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            excel_safe_dataframe(frame).to_excel(
                writer,
                sheet_name=sheet_name[:31],
                index=False,
            )
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# 2. HEADER DAN VALIDASI INPUT
# ============================================================

st.title("Analisis Narasi Sentence-BERT")
st.caption(
    "Analisis kemiripan semantik, repetisi narasi, pola temporal, "
    "dan hubungan lintas akun."
)
st.divider()

if "sentiment_df" not in st.session_state:
    st.warning(
        "Hasil prediksi IndoBERT belum tersedia. "
        "Jalankan upload, preprocessing, dan analisis sentimen terlebih dahulu."
    )
    st.stop()

sentiment_df = st.session_state["sentiment_df"].copy()

required_columns = [
    "id_str", "user_id_str", "created_at", "clean_text"
]
missing_columns = [
    column for column in required_columns
    if column not in sentiment_df.columns
]

if missing_columns:
    st.error(f"Kolom wajib tidak ditemukan: {missing_columns}")
    st.stop()

if not any(
    column in sentiment_df.columns
    for column in ["sentimen_indobert", "sentimen_analisis", "sentimen_final"]
):
    st.error("Kolom hasil sentimen tidak ditemukan. Jalankan IndoBERT terlebih dahulu.")
    st.stop()

has_relevance = any(
    column in sentiment_df.columns
    for column in ["relevansi_validasi", "relevansi", "status_analisis_sentimen"]
)

if has_relevance:
    st.info(
        "Kolom relevansi/status tersedia. Baris yang secara eksplisit ditandai "
        "tidak relevan akan dikeluarkan dari analisis."
    )
else:
    st.info(
        "Dataset tidak memiliki kolom relevansi. Semua unggahan dengan teks "
        "tidak kosong akan dianalisis. Website tidak mewajibkan workbook anotasi manual."
    )

c1, c2, c3 = st.columns(3)
c1.metric("Dataset tersedia", f"{len(sentiment_df):,}")
c2.metric(
    "Teks tidak kosong",
    f"{sentiment_df['clean_text'].fillna('').astype(str).str.strip().ne('').sum():,}"
)
c3.metric("Model", "Sentence-BERT")

st.divider()

# ============================================================
# 3. PARAMETER PENELITIAN
# ============================================================

st.subheader("Parameter Analisis")
parameter_df = pd.DataFrame([
    {"Parameter": "Model", "Nilai": SENTENCE_BERT_MODEL},
    {"Parameter": "Batch size embedding", "Nilai": 64},
    {"Parameter": "Block size pencarian pasangan", "Nilai": 128},
    {"Parameter": "Threshold pasangan awal", "Nilai": "≥ 0,80"},
    {"Parameter": "Threshold pasangan terpilih", "Nilai": "≥ 0,90"},
    {"Parameter": "Batas waktu pasangan terpilih", "Nilai": "≤ 60 menit"},
    {"Parameter": "Minimal posting per akun", "Nilai": "≥ 4"},
    {"Parameter": "Minimal pasangan lintas akun", "Nilai": "≥ 2"},
    {"Parameter": "Rata-rata similarity akun", "Nilai": "≥ 0,90"},
    {"Parameter": "Rata-rata selisih waktu akun", "Nilai": "≤ 10 menit"},
    {"Parameter": "Minimal indikator kandidat", "Nilai": "3 dari 4"},
])
with st.expander("Lihat parameter yang digunakan", expanded=False):
    st.dataframe(parameter_df, use_container_width=True, hide_index=True)

st.caption(
    "Sentimen tidak digunakan sebagai indikator seleksi kandidat. "
    "Sentimen hanya menjadi informasi pendukung setelah kandidat diperoleh."
)

# ============================================================
# 4. JALANKAN ANALISIS NARASI
# ============================================================

st.header("Jalankan Analisis Narasi")
st.write(
    "Sistem membentuk embedding untuk setiap teks, mencari pasangan dengan "
    "kemiripan semantik, menerapkan filter lintas akun dan waktu, lalu "
    "menghitung empat indikator aktivitas akun."
)

if st.button(
    "Jalankan Analisis Sentence-BERT",
    type="primary",
    use_container_width=True,
    key="btn_narrative_analysis",
):
    try:
        with st.spinner("Memuat model Sentence-BERT..."):
            model, device, model_source = load_sentence_bert_model()

        st.info(f"Model: {model_source} | Perangkat: {device}")
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def update_progress(stage, current, total, message):
            fraction = current / total if total else 0.0
            if stage == "prepare":
                overall = 0.02 + 0.02 * fraction
            elif stage == "embedding":
                overall = 0.05 + 0.40 * fraction
            elif stage == "pairs":
                overall = 0.48 + 0.42 * fraction
            elif stage == "complete":
                overall = 1.0
            else:
                overall = 0.93
            progress_bar.progress(min(max(overall, 0.0), 1.0))
            progress_text.text(message)

        with st.spinner("Menjalankan analisis narasi. Proses pencarian pasangan dapat memerlukan waktu."):
            narrative_results = run_narrative_analysis(
                df=sentiment_df,
                model=model,
                progress_callback=update_progress,
            )

        st.session_state["narrative_results"] = narrative_results
        progress_bar.progress(1.0)
        progress_text.success("Analisis narasi selesai.")
        st.success("Sentence-BERT berhasil menyelesaikan analisis.")

    except Exception as error:
        st.error(f"Analisis narasi gagal: {error}")
        st.exception(error)


# ============================================================
# 5. TAMPILKAN HASIL
# ============================================================

if "narrative_results" not in st.session_state:
    st.info("Klik tombol di atas untuk menjalankan analisis narasi.")
    st.stop()

results = st.session_state["narrative_results"]
summary = results["summary"]

st.divider()
st.header("Hasil Analisis Narasi")

# KPI utama
k1, k2, k3, k4 = st.columns(4)
k1.metric("Data dianalisis", f"{summary['jumlah_data_analisis']:,}")
k2.metric("Akun dievaluasi", f"{summary['jumlah_akun_dievaluasi']:,}")
k3.metric("Pasangan awal ≥ 0,80", f"{summary['pasangan_awal_similarity_080']:,}")
k4.metric("Pasangan terpilih", f"{summary['pasangan_terpilih_similarity_090_waktu_60']:,}")

k5, k6, k7, k8 = st.columns(4)
k5.metric("Pasangan lintas akun awal", f"{summary['pasangan_awal_lintas_akun']:,}")
k6.metric("Lintas akun ≥ 0,90", f"{summary['pasangan_lintas_akun_similarity_090']:,}")
k7.metric("Kandidat akun", f"{summary['jumlah_akun_kandidat']:,}")
k8.metric("Hubungan akun unik", f"{summary['jumlah_hubungan_akun_unik']:,}")

if not summary["kolom_relevansi_tersedia"]:
    st.warning(
        "Dataset tidak menyediakan label relevansi. Karena itu, semua teks "
        "yang tidak kosong dianalisis."
        
    )

# Tabs hasil
tab_summary, tab_pairs, tab_candidates, tab_network = st.tabs([
    "Ringkasan", "Pasangan Narasi", "Kandidat Akun", "Hubungan Akun"
])

with tab_summary:
    st.subheader("Ringkasan Pemrosesan Data")
    summary_table = pd.DataFrame([
        {"Komponen": key, "Jumlah/Nilai": value}
        for key, value in summary.items()
    ])
    st.dataframe(summary_table, use_container_width=True, hide_index=True)

    st.subheader("Distribusi Sentimen pada Data Analisis")
    sentiment_counts = (
        results["df_analysis"]["sentimen_analisis"]
        .value_counts()
        .reindex(["negatif", "netral", "positif"], fill_value=0)
    )
    sentiment_chart = sentiment_counts.rename_axis("Sentimen").to_frame("Jumlah")
    st.bar_chart(sentiment_chart)

    st.subheader("Distribusi Selisih Waktu Pasangan Terpilih")
    temporal = results["df_distribusi_temporal"]
    if temporal.empty:
        st.info("Belum ada pasangan yang memenuhi kriteria waktu dan similarity.")
    else:
        st.bar_chart(temporal.set_index("kategori_waktu")[["jumlah_pasangan"]])
        st.dataframe(temporal, use_container_width=True, hide_index=True)

    st.subheader("Analisis Sensitivitas Jumlah Indikator")
    st.dataframe(
        results["df_sensitivitas"],
        use_container_width=True,
        hide_index=True,
    )

with tab_pairs:
    st.subheader("Pasangan Unggahan Terpilih")
    st.caption(
        "Pasangan harus berasal dari akun berbeda, memiliki Cosine Similarity ≥ 0,90, "
        "dan selisih waktu publikasi ≤ 60 menit."
    )
    pairs = results["df_sentimen_pasangan"]
    pair_columns = [
        "id_str_1", "id_str_2",
        "user_id_str_1", "user_id_str_2",
        "cosine_similarity", "selisih_waktu_menit",
        "sentimen_analisis_1", "sentimen_analisis_2",
        "keselarasan_sentimen", "clean_text_1", "clean_text_2",
    ]
    pair_columns = [column for column in pair_columns if column in pairs.columns]

    if pairs.empty:
        st.info("Tidak ada pasangan yang memenuhi kriteria.")
    else:
        aligned = int(pairs["keselarasan_sentimen"].eq("Selaras").sum())
        different = int(pairs["keselarasan_sentimen"].eq("Berbeda").sum())
        p1, p2, p3 = st.columns(3)
        p1.metric("Total pasangan", f"{len(pairs):,}")
        p2.metric("Sentimen selaras", f"{aligned:,}")
        p3.metric("Sentimen berbeda", f"{different:,}")
        st.dataframe(
            pairs[pair_columns],
            use_container_width=True,
            hide_index=True,
        )

with tab_candidates:
    st.subheader("Kandidat Akun dengan Indikasi Aktivitas Terkoordinasi")
    st.caption(
        "Kandidat dipilih berdasarkan minimal 3 dari 4 indikator. "
        "Status ini menunjukkan indikasi, bukan bukti bahwa akun merupakan buzzer."
    )
    candidates = results["df_kandidat_final"]
    candidate_columns = [
        "user_id", "jumlah_posting", "jumlah_pasangan_lintas_akun",
        "rata_similarity", "rata_selisih_waktu_menit",
        "indikator_posting", "indikator_lintas_akun",
        "indikator_similarity", "indikator_temporal",
        "jumlah_indikator", "sentimen_dominan",
        "jumlah_negatif", "jumlah_netral", "jumlah_positif",
    ]
    candidate_columns = [column for column in candidate_columns if column in candidates.columns]

    if candidates.empty:
        st.info("Belum ada akun yang memenuhi minimal tiga indikator.")
    else:
        st.metric("Jumlah Kandidat", f"{len(candidates):,}")
        st.dataframe(
            candidates[candidate_columns],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Distribusi Sentimen Dominan Kandidat")
    sentiment_summary = results["df_ringkasan_sentimen"]
    if sentiment_summary.empty:
        st.info("Tidak ada sentimen berlabel pada posting kandidat.")
    else:
        dominant_counts = (
            sentiment_summary["sentimen_dominan"]
            .value_counts()
            .rename_axis("Sentimen Dominan")
            .to_frame("Jumlah Akun")
        )
        st.bar_chart(dominant_counts)

with tab_network:
    st.subheader("Hubungan Antar Akun")
    relations = results["df_hubungan_akun"]
    account_map = results["df_pemetaan_akun"]

    if relations.empty:
        st.info("Belum ada hubungan akun yang dapat divisualisasikan.")
    else:
        st.dataframe(relations, use_container_width=True, hide_index=True)

        graph = nx.Graph()
        for _, row in relations.iterrows():
            graph.add_edge(
                str(row["akun_a"]),
                str(row["akun_b"]),
                weight=float(row["jumlah_pasangan"]),
                similarity=float(row["rata_similarity"]),
            )

        candidate_ids = set(
            results["df_kandidat_final"]["user_id"].astype(str)
        )
        labels = dict(
            zip(account_map["user_id"].astype(str), account_map["label_akun"].astype(str))
        )
        node_labels = {
            node: labels.get(node, node[-6:])
            for node in graph.nodes
        }
        node_sizes = [
            900 if node in candidate_ids else 500
            for node in graph.nodes
        ]

        fig, ax = plt.subplots(figsize=(12, 8))
        positions = nx.spring_layout(graph, seed=42, k=0.9)
        nx.draw_networkx_edges(
            graph, positions, ax=ax,
            width=1.5, alpha=0.65,
        )
        nx.draw_networkx_nodes(
            graph, positions, ax=ax,
            node_size=node_sizes,
        )
        nx.draw_networkx_labels(
            graph, positions, labels=node_labels,
            font_size=8, ax=ax,
        )
        ax.set_title("Relasi Akun Berdasarkan Pasangan Unggahan Terpilih")
        ax.axis("off")
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.caption(
            "Label A01, A02, dan seterusnya merupakan kode visual akun. "
            "Detail ID akun tersedia pada tabel hubungan dan pemetaan akun."
        )

# ============================================================
# 6. EKSPOR HASIL ANALISIS
# ============================================================

st.divider()
st.header("Ekspor Hasil Analisis Narasi")

try:
    excel_data = make_excel_export(results)
    st.download_button(
        label="Download Hasil Analisis Narasi (Excel)",
        data=excel_data,
        file_name="hasil_analisis_narasi_MBG.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )
except Exception as error:
    st.error(f"Gagal menyiapkan file Excel: {error}")
