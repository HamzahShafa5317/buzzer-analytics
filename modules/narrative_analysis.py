import re
import unicodedata

import numpy as np
import pandas as pd

from config import (
    SENTENCE_BERT_MODEL,
    SENTENCE_BERT_DIR,
    EMBEDDING_DIMENSION,
    INITIAL_SIMILARITY_THRESHOLD,
    PAIR_SIMILARITY_THRESHOLD,
    MAX_PAIR_TIME_MINUTES,
    MIN_POSTS,
    MIN_CROSS_ACCOUNT_PAIRS,
    MIN_ACCOUNT_SIMILARITY,
    MAX_ACCOUNT_TIME_MINUTES,
    MIN_CANDIDATE_INDICATORS,
)

LABEL_SENTIMEN = ["negatif", "netral", "positif"]
BATCH_SIZE = 64
BLOCK_SIZE = 128


def _notify(callback, stage, current, total, message):
    if callback is not None:
        callback(stage, int(current), int(total), message)


def _normalize_text_key(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    value = value.casefold()
    return re.sub(r"\s+", " ", value).strip()


def prepare_narrative_dataframe(df):
    """Menyiapkan data sesuai praproses pada notebook Sentence-BERT.

    Jika kolom relevansi tersedia, baris berlabel 'tidak' dikeluarkan.
    Jika tidak tersedia, seluruh unggahan nonkosong digunakan.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input analisis narasi harus berupa pandas DataFrame.")

    data = df.copy()
    required = ["id_str", "user_id_str", "created_at", "clean_text"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Kolom wajib analisis narasi tidak ditemukan: {missing}")

    if "full_text" not in data.columns:
        data["full_text"] = data["clean_text"]

    if "sentimen_indobert" in data.columns:
        sentiment_source = "sentimen_indobert"
    elif "sentimen_analisis" in data.columns:
        sentiment_source = "sentimen_analisis"
    elif "sentimen_final" in data.columns:
        sentiment_source = "sentimen_final"
    else:
        raise ValueError(
            "Kolom sentimen tidak ditemukan. Jalankan analisis IndoBERT terlebih dahulu."
        )

    data["id_str"] = data["id_str"].astype("string").str.strip()
    data["user_id_str"] = data["user_id_str"].astype("string").str.strip()
    data["clean_text"] = data["clean_text"].fillna("").astype(str).str.strip()
    data["full_text"] = data["full_text"].fillna("").astype(str)
    data["sentimen_analisis"] = (
        data[sentiment_source].astype("string").str.strip().str.lower()
    )

    if "sumber_label" not in data.columns:
        data["sumber_label"] = "IndoBERT"

    # Validasi ID dan waktu dilakukan sebelum filtering, agar sumber bermasalah terlihat.
    if data["id_str"].isna().any() or data["id_str"].eq("").any():
        raise ValueError("Terdapat ID posting kosong pada data yang diunggah.")
    if data["user_id_str"].isna().any() or data["user_id_str"].eq("").any():
        raise ValueError("Terdapat ID akun kosong pada data yang diunggah.")
    if data["id_str"].duplicated().any():
        raise ValueError(
            "ID posting duplikat ditemukan. Pastikan deduplikasi berdasarkan id_str telah dijalankan."
        )

    created_utc = pd.to_datetime(data["created_at"], utc=True, errors="coerce")
    if created_utc.isna().any():
        bad_count = int(created_utc.isna().sum())
        raise ValueError(f"Terdapat {bad_count} waktu publikasi yang tidak valid.")

    data["created_at_utc"] = created_utc
    data["created_at_wib"] = created_utc.dt.tz_convert("Asia/Jakarta")

    # Filter relevansi opsional: tidak memaksa pengguna menyediakan anotasi manual.
    irrelevant_mask = pd.Series(False, index=data.index)
    relevance_column = None

    if "status_analisis_sentimen" in data.columns:
        status = data["status_analisis_sentimen"].astype("string").str.strip().str.lower()
        irrelevant_mask |= status.eq("dikeluarkan_tidak_relevan").fillna(False)

    for candidate in ["relevansi_validasi", "relevansi"]:
        if candidate in data.columns:
            relevance_column = candidate
            relevance = data[candidate].astype("string").str.strip().str.lower()
            irrelevant_mask |= relevance.isin(["tidak", "tidak relevan", "irrelevant", "no"])
            break

    text_empty_mask = data["clean_text"].eq("")
    data_used = data.loc[~irrelevant_mask & ~text_empty_mask].copy()
    data_used = data_used.reset_index(drop=True)

    if data_used.empty:
        raise ValueError("Tidak ada unggahan relevan dan tidak kosong untuk dianalisis.")

    # Sentiment labels may be unavailable for some rows; narrative similarity still works.
    data_used["sentimen_analisis"] = data_used["sentimen_analisis"].where(
        data_used["sentimen_analisis"].isin(LABEL_SENTIMEN), pd.NA
    )

    summary = {
        "jumlah_data_input": len(data),
        "jumlah_tidak_relevan_dikeluarkan": int(irrelevant_mask.sum()),
        "jumlah_teks_kosong_dikeluarkan": int((~irrelevant_mask & text_empty_mask).sum()),
        "jumlah_data_analisis": len(data_used),
        "jumlah_akun_analisis": int(data_used["user_id_str"].nunique()),
        "kolom_relevansi_tersedia": relevance_column is not None,
        "kolom_sentimen_digunakan": sentiment_source,
    }
    return data_used, summary


def _encode_texts(model, texts, progress_callback=None, batch_size=BATCH_SIZE):
    chunks = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        encoded = model.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        chunks.append(np.asarray(encoded, dtype=np.float32))
        _notify(
            progress_callback,
            "embedding",
            end,
            total,
            f"Membentuk embedding {end:,}/{total:,} unggahan",
        )
    embeddings = np.vstack(chunks).astype(np.float32, copy=False)
    if len(embeddings) != total:
        raise ValueError("Jumlah embedding tidak sama dengan jumlah teks.")
    if embeddings.ndim != 2:
        raise ValueError("Dimensi embedding tidak valid.")
    if EMBEDDING_DIMENSION and embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Dimensi embedding {embeddings.shape[1]}, sedangkan konfigurasi mengharapkan {EMBEDDING_DIMENSION}."
        )
    return embeddings


def _build_initial_pairs(embeddings, progress_callback=None, block_size=BLOCK_SIZE):
    n_rows = len(embeddings)
    pair_records = []
    total_blocks = (n_rows + block_size - 1) // block_size

    for block_number, start in enumerate(range(0, n_rows, block_size), start=1):
        end = min(start + block_size, n_rows)
        score_block = embeddings[start:end] @ embeddings.T

        for offset in range(end - start):
            i = start + offset
            if i + 1 >= n_rows:
                continue
            scores = score_block[offset]
            candidates = (
                np.flatnonzero(scores[i + 1:] >= INITIAL_SIMILARITY_THRESHOLD)
                + i + 1
            )
            if len(candidates):
                pair_records.extend(
                    (i, int(j), float(scores[j]))
                    for j in candidates
                )

        _notify(
            progress_callback,
            "pairs",
            block_number,
            total_blocks,
            f"Mencari pasangan similarity: blok {block_number}/{total_blocks}",
        )
        del score_block

    return pd.DataFrame(
        pair_records,
        columns=["idx_1", "idx_2", "cosine_similarity"],
    )


def _complete_pair_metadata(df_analysis, df_pairs_initial):
    pair_columns = [
        "id_str",
        "user_id_str",
        "created_at_utc",
        "created_at_wib",
        "full_text",
        "clean_text",
        "sentimen_analisis",
        "sumber_label",
    ]
    for column in pair_columns:
        if column not in df_analysis.columns:
            df_analysis[column] = pd.NA

    if df_pairs_initial.empty:
        for column in pair_columns:
            df_pairs_initial[f"{column}_1"] = pd.Series(dtype="object")
            df_pairs_initial[f"{column}_2"] = pd.Series(dtype="object")
        df_pairs_initial["selisih_waktu_menit"] = pd.Series(dtype="float64")
        df_pairs_initial["akun_sama"] = pd.Series(dtype="bool")
        return df_pairs_initial

    idx1 = df_pairs_initial["idx_1"].to_numpy(dtype=int)
    idx2 = df_pairs_initial["idx_2"].to_numpy(dtype=int)
    left = df_analysis.iloc[idx1].reset_index(drop=True)
    right = df_analysis.iloc[idx2].reset_index(drop=True)

    for column in pair_columns:
        df_pairs_initial[f"{column}_1"] = left[column].to_numpy()
        df_pairs_initial[f"{column}_2"] = right[column].to_numpy()

    time_left = pd.to_datetime(left["created_at_utc"], utc=True)
    time_right = pd.to_datetime(right["created_at_utc"], utc=True)
    df_pairs_initial["selisih_waktu_menit"] = (
        (time_left - time_right).abs().dt.total_seconds() / 60
    )
    df_pairs_initial["akun_sama"] = (
        left["user_id_str"].astype(str).to_numpy()
        == right["user_id_str"].astype(str).to_numpy()
    )
    return df_pairs_initial


def _pairs_to_long(df_pairs, metric_name):
    if df_pairs.empty:
        return pd.DataFrame(columns=["user_id", "nilai"])

    left = df_pairs[["user_id_str_1", metric_name]].rename(
        columns={"user_id_str_1": "user_id", metric_name: "nilai"}
    )
    right = df_pairs[["user_id_str_2", metric_name]].rename(
        columns={"user_id_str_2": "user_id", metric_name: "nilai"}
    )
    return pd.concat([left, right], ignore_index=True).dropna(subset=["nilai"])


def _build_account_indicators(df_analysis, df_pairs_selected, df_base_cross, df_high_semantic_cross):
    # Indikator 1: jumlah posting per akun.
    posting_count = (
        df_analysis.groupby("user_id_str")["id_str"]
        .nunique()
        .rename("jumlah_posting")
    )

    # Indikator 2: banyaknya pasangan akhir yang melibatkan akun.
    if df_pairs_selected.empty:
        pair_count = pd.Series(dtype="int64", name="jumlah_pasangan_lintas_akun")
    else:
        endpoints = pd.concat(
            [df_pairs_selected["user_id_str_1"], df_pairs_selected["user_id_str_2"]],
            ignore_index=True,
        )
        pair_count = endpoints.value_counts().rename("jumlah_pasangan_lintas_akun")

    # Indikator 3: rata-rata similarity dari pool pasangan awal lintas akun.
    long_similarity = _pairs_to_long(df_base_cross, "cosine_similarity")
    if long_similarity.empty:
        mean_similarity = pd.Series(dtype="float64", name="rata_similarity")
    else:
        mean_similarity = (
            long_similarity.groupby("user_id")["nilai"]
            .mean()
            .rename("rata_similarity")
        )

    # Indikator 4: rata-rata waktu dari similarity >= 0,90 sebelum filter 60 menit.
    long_time = _pairs_to_long(df_high_semantic_cross, "selisih_waktu_menit")
    if long_time.empty:
        mean_time = pd.Series(dtype="float64", name="rata_selisih_waktu_menit")
    else:
        mean_time = (
            long_time.groupby("user_id")["nilai"]
            .mean()
            .rename("rata_selisih_waktu_menit")
        )

    indicators = (
        posting_count.reset_index()
        .rename(columns={"user_id_str": "user_id"})
    )
    indicators["user_id"] = indicators["user_id"].astype(str)
    indicators["jumlah_pasangan_lintas_akun"] = (
        indicators["user_id"].map(pair_count).fillna(0).astype(int)
    )
    indicators["rata_similarity"] = indicators["user_id"].map(mean_similarity)
    indicators["rata_selisih_waktu_menit"] = indicators["user_id"].map(mean_time)

    indicators["indikator_posting"] = (
        indicators["jumlah_posting"] >= MIN_POSTS
    ).astype(int)
    indicators["indikator_lintas_akun"] = (
        indicators["jumlah_pasangan_lintas_akun"] >= MIN_CROSS_ACCOUNT_PAIRS
    ).astype(int)
    indicators["indikator_similarity"] = (
        indicators["rata_similarity"].fillna(-np.inf) >= MIN_ACCOUNT_SIMILARITY
    ).astype(int)
    indicators["indikator_temporal"] = (
        indicators["rata_selisih_waktu_menit"].fillna(np.inf)
        <= MAX_ACCOUNT_TIME_MINUTES
    ).astype(int)

    indicator_columns = [
        "indikator_posting",
        "indikator_lintas_akun",
        "indikator_similarity",
        "indikator_temporal",
    ]
    indicators["jumlah_indikator"] = indicators[indicator_columns].sum(axis=1)
    indicators["status_identifikasi"] = np.where(
        indicators["jumlah_indikator"] >= MIN_CANDIDATE_INDICATORS,
        "Kandidat",
        "Tidak masuk kandidat",
    )

    candidates = (
        indicators.loc[
            indicators["jumlah_indikator"] >= MIN_CANDIDATE_INDICATORS
        ]
        .copy()
        .sort_values(
            ["jumlah_indikator", "jumlah_pasangan_lintas_akun"],
            ascending=[False, False],
        )
        .reset_index(drop=True)
    )
    return indicators, candidates, indicator_columns


def _build_candidate_sentiment(df_analysis, candidates):
    if candidates.empty:
        empty = pd.DataFrame(
            columns=[
                "user_id", "jumlah_negatif", "jumlah_netral", "jumlah_positif",
                "persentase_negatif", "persentase_netral", "persentase_positif",
                "sentimen_dominan", "jumlah_posting_berlabel",
            ]
        )
        return empty, candidates.copy()

    candidate_ids = set(candidates["user_id"].astype(str))
    posts = df_analysis.loc[
        df_analysis["user_id_str"].astype(str).isin(candidate_ids)
    ].copy()
    posts = posts.loc[posts["sentimen_analisis"].isin(LABEL_SENTIMEN)].copy()

    if posts.empty:
        sentiment_summary = pd.DataFrame(
            columns=[
                "user_id", "jumlah_negatif", "jumlah_netral", "jumlah_positif",
                "persentase_negatif", "persentase_netral", "persentase_positif",
                "sentimen_dominan", "jumlah_posting_berlabel",
            ]
        )
    else:
        pivot = (
            posts.groupby(["user_id_str", "sentimen_analisis"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=LABEL_SENTIMEN, fill_value=0)
        )
        pivot.columns = [
            "jumlah_negatif", "jumlah_netral", "jumlah_positif"
        ]
        pivot.index.name = "user_id"
        pivot = pivot.reset_index()
        count_columns = [
            "jumlah_negatif", "jumlah_netral", "jumlah_positif"
        ]
        pivot["jumlah_posting_berlabel"] = pivot[count_columns].sum(axis=1)
        denominator = pivot["jumlah_posting_berlabel"].replace(0, np.nan)
        pivot["persentase_negatif"] = pivot["jumlah_negatif"] / denominator * 100
        pivot["persentase_netral"] = pivot["jumlah_netral"] / denominator * 100
        pivot["persentase_positif"] = pivot["jumlah_positif"] / denominator * 100

        def dominant(row):
            values = {
                "negatif": row["jumlah_negatif"],
                "netral": row["jumlah_netral"],
                "positif": row["jumlah_positif"],
            }
            maximum = max(values.values())
            if maximum == 0:
                return "Tidak tersedia"
            winners = [label for label, value in values.items() if value == maximum]
            return winners[0] if len(winners) == 1 else "Seri (" + ", ".join(winners) + ")"

        pivot["sentimen_dominan"] = pivot.apply(dominant, axis=1)
        sentiment_summary = pivot

    candidates_final = candidates.merge(
        sentiment_summary,
        on="user_id",
        how="left",
        validate="one_to_one",
    )
    return sentiment_summary, candidates_final


def _build_pair_sentiment(df_pairs_selected):
    pairs = df_pairs_selected.copy()
    if pairs.empty:
        pairs["keselarasan_sentimen"] = pd.Series(dtype="string")
        return pairs

    left = pairs["sentimen_analisis_1"].astype("string")
    right = pairs["sentimen_analisis_2"].astype("string")
    valid = left.isin(LABEL_SENTIMEN) & right.isin(LABEL_SENTIMEN)

    pairs["keselarasan_sentimen"] = np.select(
        [~valid, left.eq(right)],
        ["Tidak tersedia", "Selaras"],
        default="Berbeda",
    )
    return pairs


def _build_account_relations(df_pairs_selected, candidates):
    candidate_ids = set(candidates["user_id"].astype(str))
    if df_pairs_selected.empty or not candidate_ids:
        relations = pd.DataFrame(
            columns=[
                "akun_a", "akun_b", "jumlah_pasangan", "rata_similarity",
                "maks_similarity", "rata_selisih_waktu_menit",
            ]
        )
    else:
        pairs = df_pairs_selected.copy()
        involved = (
            pairs["user_id_str_1"].astype(str).isin(candidate_ids)
            | pairs["user_id_str_2"].astype(str).isin(candidate_ids)
        )
        pairs = pairs.loc[involved].copy()
        if pairs.empty:
            relations = pd.DataFrame(
                columns=[
                    "akun_a", "akun_b", "jumlah_pasangan", "rata_similarity",
                    "maks_similarity", "rata_selisih_waktu_menit",
                ]
            )
        else:
            pairs["akun_a"] = pairs.apply(
                lambda row: min(str(row["user_id_str_1"]), str(row["user_id_str_2"])),
                axis=1,
            )
            pairs["akun_b"] = pairs.apply(
                lambda row: max(str(row["user_id_str_1"]), str(row["user_id_str_2"])),
                axis=1,
            )
            relations = (
                pairs.groupby(["akun_a", "akun_b"], as_index=False)
                .agg(
                    jumlah_pasangan=("cosine_similarity", "size"),
                    rata_similarity=("cosine_similarity", "mean"),
                    maks_similarity=("cosine_similarity", "max"),
                    rata_selisih_waktu_menit=("selisih_waktu_menit", "mean"),
                )
            )

    if relations.empty:
        account_map = pd.DataFrame(
            columns=["user_id", "label_akun", "status_identifikasi", "jumlah_posting"]
        )
    else:
        account_ids = sorted(
            set(relations["akun_a"]) | set(relations["akun_b"])
        )
        label_map = {
            user_id: f"A{i + 1:02d}"
            for i, user_id in enumerate(account_ids)
        }
        posting_map = (
            candidates.set_index("user_id")["jumlah_posting"].to_dict()
            if not candidates.empty else {}
        )
        account_map = pd.DataFrame({
            "user_id": account_ids,
            "label_akun": [label_map[user_id] for user_id in account_ids],
            "status_identifikasi": [
                "Kandidat" if user_id in candidate_ids else "Akun terhubung"
                for user_id in account_ids
            ],
            "jumlah_posting": [posting_map.get(user_id, 0) for user_id in account_ids],
        })

    return relations, account_map


def _build_temporal_distribution(df_pairs_selected):
    labels = [
        "≤ 1 menit", ">1–5 menit", ">5–10 menit",
        ">10–30 menit", ">30–60 menit",
    ]
    if df_pairs_selected.empty:
        return pd.DataFrame({
            "kategori_waktu": pd.Categorical([], categories=labels, ordered=True),
            "jumlah_pasangan": pd.Series(dtype="int64"),
            "persentase": pd.Series(dtype="float64"),
        })

    bins = [-np.inf, 1, 5, 10, 30, 60]
    pairs = df_pairs_selected.copy()
    pairs["kategori_waktu"] = pd.cut(
        pairs["selisih_waktu_menit"],
        bins=bins,
        labels=labels,
        right=True,
    )
    distribution = (
        pairs["kategori_waktu"]
        .value_counts(sort=False)
        .rename_axis("kategori_waktu")
        .reset_index(name="jumlah_pasangan")
    )
    total = distribution["jumlah_pasangan"].sum()
    distribution["persentase"] = (
        distribution["jumlah_pasangan"] / total * 100 if total else 0.0
    )
    return distribution


def run_narrative_analysis(df, model, progress_callback=None):
    """Jalankan pipeline Sentence-BERT dan identifikasi kandidat akun.

    Parameter mengikuti Sentence_Bert_2_Final.ipynb.
    Mengembalikan semua dataframe hasil agar dapat ditampilkan/diunduh.
    """
    _notify(progress_callback, "prepare", 0, 1, "Menyiapkan data analisis")
    df_analysis, summary = prepare_narrative_dataframe(df)
    _notify(progress_callback, "prepare", 1, 1, "Data siap dianalisis")

    texts = df_analysis["clean_text"].tolist()
    embeddings = _encode_texts(
        model,
        texts,
        progress_callback=progress_callback,
        batch_size=BATCH_SIZE,
    )

    df_pairs_initial = _build_initial_pairs(
        embeddings,
        progress_callback=progress_callback,
        block_size=BLOCK_SIZE,
    )
    df_pairs_initial = _complete_pair_metadata(
        df_analysis,
        df_pairs_initial,
    )

    if df_pairs_initial.empty:
        df_base_cross = df_pairs_initial.copy()
        df_high_semantic_cross = df_pairs_initial.copy()
        df_pairs_selected = df_pairs_initial.copy()
    else:
        # Pool pasangan lintas akun.
        df_base_cross = (
            df_pairs_initial.loc[~df_pairs_initial["akun_sama"]]
            .copy()
            .reset_index(drop=True)
        )

        # Similarity tinggi sebelum pembatasan waktu 60 menit.
        df_high_semantic_cross = (
            df_base_cross.loc[
                df_base_cross["cosine_similarity"] >= PAIR_SIMILARITY_THRESHOLD
            ]
            .copy()
            .reset_index(drop=True)
        )

        # Pasangan akhir untuk seleksi.
        df_pairs_selected = (
            df_high_semantic_cross.loc[
                df_high_semantic_cross["selisih_waktu_menit"] <= MAX_PAIR_TIME_MINUTES
            ]
            .copy()
            .reset_index(drop=True)
        )

    indicators, candidates, indicator_columns = _build_account_indicators(
        df_analysis,
        df_pairs_selected,
        df_base_cross,
        df_high_semantic_cross,
    )

    sentiment_summary, candidates_final = _build_candidate_sentiment(
        df_analysis,
        candidates,
    )

    pair_sentiment = _build_pair_sentiment(
        df_pairs_selected,
    )

    relations, account_map = _build_account_relations(
        df_pairs_selected,
        candidates_final,
    )

    temporal_distribution = _build_temporal_distribution(
        df_pairs_selected,
    )

    sensitivity_rows = []
    for minimum in [2, 3, 4]:
        subset = indicators.loc[indicators["jumlah_indikator"] >= minimum]
        sensitivity_rows.append({
            "minimal_indikator": minimum,
            "jumlah_akun_kandidat": len(subset),
            "jumlah_akun_dievaluasi": len(indicators),
            "persentase_kandidat": (
                len(subset) / len(indicators) * 100 if len(indicators) else 0.0
            ),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    # Validasi hasil seperti pada notebook.
    if not df_pairs_selected.empty:
        assert (df_pairs_selected["cosine_similarity"] >= PAIR_SIMILARITY_THRESHOLD).all()
        assert (df_pairs_selected["selisih_waktu_menit"] <= MAX_PAIR_TIME_MINUTES).all()
        assert (df_pairs_selected["user_id_str_1"] != df_pairs_selected["user_id_str_2"]).all()

    if not indicators.empty:
        assert indicators[indicator_columns].isin([0, 1]).all().all()
    if not candidates_final.empty:
        assert (candidates_final["jumlah_indikator"] >= MIN_CANDIDATE_INDICATORS).all()
        assert candidates_final["user_id"].is_unique
        assert (candidates_final["jumlah_indikator"] <= 4).all()

    summary.update({
        "model_sentence_bert": SENTENCE_BERT_MODEL,
        "dimensi_embedding": int(embeddings.shape[1]),
        "jumlah_embedding": int(len(embeddings)),
        "pasangan_awal_similarity_080": int(len(df_pairs_initial)),
        "pasangan_awal_lintas_akun": int(len(df_base_cross)),
        "pasangan_lintas_akun_similarity_090": int(len(df_high_semantic_cross)),
        "pasangan_terpilih_similarity_090_waktu_60": int(len(df_pairs_selected)),
        "jumlah_akun_dievaluasi": int(len(indicators)),
        "jumlah_akun_kandidat": int(len(candidates_final)),
        "jumlah_hubungan_akun_unik": int(len(relations)),
        "jumlah_pasangan_sentimen_selaras": int(
            pair_sentiment["keselarasan_sentimen"].eq("Selaras").sum()
        ) if not pair_sentiment.empty else 0,
        "jumlah_pasangan_sentimen_berbeda": int(
            pair_sentiment["keselarasan_sentimen"].eq("Berbeda").sum()
        ) if not pair_sentiment.empty else 0,
    })

    _notify(
        progress_callback,
        "complete",
        1,
        1,
        "Analisis narasi dan identifikasi kandidat selesai",
    )

    return {
        "df_analysis": df_analysis,
        "df_pairs_initial": df_pairs_initial,
        "df_base_cross": df_base_cross,
        "df_high_semantic_cross": df_high_semantic_cross,
        "df_pairs_selected": df_pairs_selected,
        "df_indikator": indicators,
        "df_kandidat": candidates,
        "df_ringkasan_sentimen": sentiment_summary,
        "df_kandidat_final": candidates_final,
        "df_sentimen_pasangan": pair_sentiment,
        "df_hubungan_akun": relations,
        "df_pemetaan_akun": account_map,
        "df_distribusi_temporal": temporal_distribution,
        "df_sensitivitas": sensitivity,
        "summary": summary,
    }
