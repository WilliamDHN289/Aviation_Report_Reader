"""
FAA/NTSB narrative clustering pipeline.

Stages implemented:
1) Outcome language stripping (regex)
2) Embedding with sentence-transformers/all-MiniLM-L6-v2
3a) UMAP (10D for clustering + 2D for visualization)
3b) HDBSCAN clustering
4) Optional GPT labeling scaffold (OpenAI API)
5) Expert correction artifacts (review templates and files)

Usage example:
python FAA_pipeline/pipeline.py \
  --input-csv NLP_pipeline/ntsb_ml_filled_1.csv \
  --output-dir FAA_pipeline/output \
  --state-col State \
  --text-col clean_text \
  --run-gpt-labeling
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("faa_pipeline")


# Stage 1: Regex outcome language stripping
OUTCOME_PATTERNS = [
    r"\bfatal(?:ity|ities)?\b",
    r"\binjur(?:y|ies|ed)\b",
    r"\bdamage(?:d)?\b",
    r"\bcrash(?:ed|es)?\b",
    r"\bdestroy(?:ed|s)?\b",
]
OUTCOME_REGEX = re.compile("|".join(OUTCOME_PATTERNS), flags=re.IGNORECASE)


# Stage 4 prompt: ban generic "pilot error" labels
LABELING_SYSTEM_PROMPT = """You are an aviation safety analyst.
Given 8-10 sample narratives from one cluster, generate:
1) a concise cluster label (mechanism-specific),
2) a broader cause theme label,
3) one-sentence rationale.

Rules:
- DO NOT use generic labels like "pilot error".
- Name specific mechanism (e.g., "gear extension checklist omission",
  "tailwheel directional control loss in crosswind landing").
- Keep labels short and operationally actionable.
Return strict JSON with keys:
cluster_label, theme_label, rationale.
"""


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def strip_outcome_language(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = OUTCOME_REGEX.sub(" ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def detect_text_column(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested and requested in df.columns:
        return requested
    candidates = ["clean_text", "fulltext", "narrative", "Narrative", "text"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not detect narrative text column. Pass --text-col explicitly."
    )


@dataclass
class StateParams:
    n_neighbors: int
    min_cluster_size: int
    min_samples: int


def params_for_state(state_name: str, n_rows: int) -> StateParams:
    s = (state_name or "").strip().lower()
    if s in {"alaska", "ak"}:
        return StateParams(n_neighbors=10, min_cluster_size=3, min_samples=2)
    if n_rows >= 500:
        return StateParams(n_neighbors=15, min_cluster_size=5, min_samples=3)
    return StateParams(n_neighbors=12, min_cluster_size=3, min_samples=2)


def load_dependencies():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        import umap  # noqa: F401
        import hdbscan  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Missing dependencies. Install:\n"
            "pip install sentence-transformers umap-learn hdbscan scikit-learn matplotlib"
        ) from exc


def generate_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    LOGGER.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb


def run_umap_and_hdbscan(
    embeddings: np.ndarray,
    params: StateParams,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import umap
    import hdbscan

    reducer_10d = umap.UMAP(
        n_neighbors=params.n_neighbors,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    reducer_2d = umap.UMAP(
        n_neighbors=params.n_neighbors,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )

    emb_10d = reducer_10d.fit_transform(embeddings)
    emb_2d = reducer_2d.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(emb_10d)
    probs = getattr(clusterer, "probabilities_", np.zeros(len(labels)))
    return emb_10d, emb_2d, labels, probs


def summarize_clusters(df_state: pd.DataFrame) -> pd.DataFrame:
    total = len(df_state)
    clustered = int((df_state["cluster_id"] != -1).sum())
    noise = int((df_state["cluster_id"] == -1).sum())
    cluster_count = int(df_state.loc[df_state["cluster_id"] != -1, "cluster_id"].nunique())
    return pd.DataFrame(
        [
            {
                "total_records": total,
                "clustered_records": clustered,
                "noise_records": noise,
                "noise_rate": round(noise / total, 4) if total else 0.0,
                "clusters": cluster_count,
            }
        ]
    )


def save_scatter_plot(df_state: pd.DataFrame, out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df_state["umap_x"],
        df_state["umap_y"],
        c=df_state["cluster_id"],
        s=12,
        alpha=0.75,
        cmap="tab20",
    )
    plt.colorbar(scatter, label="Cluster ID (-1 = noise)")
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def sample_cluster_texts(
    df_clustered: pd.DataFrame,
    text_col: str,
    sample_size: int = 10,
    seed: int = 42,
) -> Dict[int, List[str]]:
    rng = random.Random(seed)
    out: Dict[int, List[str]] = {}
    for cid, grp in df_clustered.groupby("cluster_id"):
        if cid == -1:
            continue
        texts = grp[text_col].dropna().astype(str).tolist()
        if not texts:
            out[cid] = []
            continue
        k = min(sample_size, len(texts))
        out[cid] = rng.sample(texts, k=k)
    return out


def _compact_text(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_prompt_payload(
    cid: int,
    samples: List[str],
    sample_cap: int,
    sample_char_cap: int,
    prompt_char_cap: int,
) -> Dict[str, object]:
    trimmed = [_compact_text(s, sample_char_cap) for s in samples[:sample_cap]]
    payload = {"cluster_id": cid, "samples": trimmed}
    # Keep shrinking until prompt JSON fits char budget.
    while len(json.dumps(payload)) > prompt_char_cap and len(trimmed) > 1:
        trimmed = trimmed[:-1]
        payload = {"cluster_id": cid, "samples": trimmed}
    return payload


def run_gpt_labeling(
    samples_by_cluster: Dict[int, List[str]],
    model: str,
    sample_cap: int = 6,
    sample_char_cap: int = 700,
    prompt_char_cap: int = 5000,
    max_retries: int = 4,
) -> pd.DataFrame:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise ImportError(
            "OpenAI SDK missing. Install with: pip install openai"
        ) from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set; cannot run GPT labeling.")

    client = OpenAI(api_key=api_key)
    rows = []
    for cid, samples in samples_by_cluster.items():
        if not samples:
            rows.append(
                {
                    "cluster_id": cid,
                    "cluster_label": "NEEDS_REVIEW",
                    "theme_label": "NEEDS_REVIEW",
                    "rationale": "No sample text available.",
                }
            )
            continue
        response = None
        used_payload = None
        local_sample_cap = max(1, sample_cap)
        local_sample_char_cap = max(200, sample_char_cap)

        for attempt in range(max_retries):
            used_payload = _build_prompt_payload(
                cid=cid,
                samples=samples,
                sample_cap=local_sample_cap,
                sample_char_cap=local_sample_char_cap,
                prompt_char_cap=prompt_char_cap,
            )
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": LABELING_SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(used_payload)},
                    ],
                )
                break
            except Exception as exc:
                msg = str(exc).lower()
                is_tpm_or_size = (
                    "rate_limit_exceeded" in msg
                    or "request too large" in msg
                    or "tokens per min" in msg
                    or "429" in msg
                )
                if attempt == max_retries - 1 or not is_tpm_or_size:
                    raise
                # Back off and shrink payload for next try.
                LOGGER.warning(
                    "GPT labeling retry for cluster=%s (attempt %s/%s). "
                    "Shrinking payload due to rate/token limit.",
                    cid,
                    attempt + 1,
                    max_retries,
                )
                local_sample_cap = max(1, local_sample_cap - 1)
                local_sample_char_cap = max(200, int(local_sample_char_cap * 0.7))
                time.sleep(2 + attempt * 2)

        if response is None:
            rows.append(
                {
                    "cluster_id": cid,
                    "cluster_label": "RATE_LIMIT_ERROR",
                    "theme_label": "RATE_LIMIT_ERROR",
                    "rationale": "Labeling failed after retries due to token/rate limits.",
                }
            )
            continue

        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = {
                "cluster_label": "PARSE_ERROR",
                "theme_label": "PARSE_ERROR",
                "rationale": content[:500],
            }
        rows.append(
            {
                "cluster_id": cid,
                "cluster_label": parsed.get("cluster_label", "NEEDS_REVIEW"),
                "theme_label": parsed.get("theme_label", "NEEDS_REVIEW"),
                "rationale": parsed.get("rationale", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster_id")


def build_expert_review_template(
    state: str,
    summary_df: pd.DataFrame,
    cluster_counts: pd.DataFrame,
    out_md: Path,
) -> None:
    summary = summary_df.iloc[0].to_dict()
    lines = [
        f"# Expert Correction Notes - {state}",
        "",
        "## Dataset Snapshot",
        f"- Total records: {summary['total_records']}",
        f"- Clustered records: {summary['clustered_records']}",
        f"- Noise records: {summary['noise_records']}",
        f"- Noise rate: {summary['noise_rate']}",
        f"- Cluster count: {summary['clusters']}",
        "",
        "## Reviewer Instructions",
        "1. Validate each cluster's mechanism-level label (avoid generic 'pilot error').",
        "2. Merge/split clusters if narratives show mixed mechanisms.",
        "3. Move clusters across themes where needed.",
        "4. Add data quality flags (e.g., sparse, mixed narratives, suspected duplicates).",
        "5. Document final theme map and rationale.",
        "",
        "## Cluster Worklist",
        "",
    ]
    for _, row in cluster_counts.iterrows():
        lines.extend(
            [
                f"### Cluster {int(row['cluster_id'])} ({int(row['count'])} records)",
                "- Proposed label:",
                "- Final label:",
                "- Theme:",
                "- Decision notes:",
                "- Data quality flags:",
                "",
            ]
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")


def process_state(
    state: str,
    df_state: pd.DataFrame,
    text_col: str,
    out_dir: Path,
    embedding_model: str,
    run_gpt: bool,
    gpt_model: str,
    gpt_sample_cap: int,
    gpt_sample_char_cap: int,
    gpt_prompt_char_cap: int,
    seed: int,
) -> pd.DataFrame:
    state_slug = re.sub(r"[^a-zA-Z0-9]+", "_", state.strip()).strip("_").lower()
    state_dir = out_dir / state_slug
    state_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Processing state=%s, n=%d", state, len(df_state))
    params = params_for_state(state, len(df_state))
    LOGGER.info("Using params: %s", params)

    texts = df_state[text_col].fillna("").astype(str).tolist()
    embeddings = generate_embeddings(texts, model_name=embedding_model)
    emb_10d, emb_2d, labels, probs = run_umap_and_hdbscan(embeddings, params=params)

    out_df = df_state.copy()
    out_df["cluster_id"] = labels
    out_df["cluster_probability"] = probs
    out_df["umap_x"] = emb_2d[:, 0]
    out_df["umap_y"] = emb_2d[:, 1]

    for i in range(emb_10d.shape[1]):
        out_df[f"umap10_{i}"] = emb_10d[:, i]

    clustered_csv = state_dir / "clustered_records.csv"
    out_df.to_csv(clustered_csv, index=False)

    summary_df = summarize_clusters(out_df)
    summary_df.to_csv(state_dir / "summary.csv", index=False)

    cluster_counts = (
        out_df.loc[out_df["cluster_id"] != -1, "cluster_id"]
        .value_counts()
        .rename_axis("cluster_id")
        .reset_index(name="count")
        .sort_values("cluster_id")
    )
    cluster_counts.to_csv(state_dir / "cluster_counts.csv", index=False)

    save_scatter_plot(
        out_df,
        state_dir / "clusters_umap2d.png",
        title=f"{state}: UMAP + HDBSCAN Clusters",
    )

    sample_df_rows = []
    samples_by_cluster = sample_cluster_texts(
        out_df[out_df["cluster_id"] != -1],
        text_col=text_col,
        sample_size=10,
        seed=seed,
    )
    for cid, samples in samples_by_cluster.items():
        for idx, txt in enumerate(samples):
            sample_df_rows.append(
                {"cluster_id": cid, "sample_index": idx, "sample_text": txt}
            )
    pd.DataFrame(sample_df_rows).to_csv(state_dir / "cluster_samples.csv", index=False)

    if run_gpt:
        labels_df = run_gpt_labeling(
            samples_by_cluster,
            model=gpt_model,
            sample_cap=gpt_sample_cap,
            sample_char_cap=gpt_sample_char_cap,
            prompt_char_cap=gpt_prompt_char_cap,
        )
        labels_df.to_csv(state_dir / "gpt_cluster_labels.csv", index=False)
    else:
        pd.DataFrame(
            [
                {
                    "cluster_id": cid,
                    "cluster_label": "PENDING_GPT_OR_MANUAL",
                    "theme_label": "PENDING_GPT_OR_MANUAL",
                    "rationale": "",
                }
                for cid in sorted(samples_by_cluster.keys())
            ]
        ).to_csv(state_dir / "gpt_cluster_labels.csv", index=False)

    build_expert_review_template(
        state=state,
        summary_df=summary_df,
        cluster_counts=cluster_counts,
        out_md=state_dir / "expert_review_notes.md",
    )
    return summary_df.assign(state=state)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FAA narrative clustering pipeline")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--state-col", type=str, default="State", help="State column")
    parser.add_argument("--text-col", type=str, default=None, help="Narrative text column")
    parser.add_argument("--id-col", type=str, default="NtsbNo", help="Record ID column")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default="gpt-4o",
        help="Model name for Stage 4 labeling",
    )
    parser.add_argument(
        "--run-gpt-labeling",
        action="store_true",
        help="Enable Stage 4 GPT labeling (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--gpt-sample-cap",
        type=int,
        default=6,
        help="Max narratives sent to GPT per cluster (default: 6)",
    )
    parser.add_argument(
        "--gpt-sample-char-cap",
        type=int,
        default=700,
        help="Max chars per narrative sent to GPT (default: 700)",
    )
    parser.add_argument(
        "--gpt-prompt-char-cap",
        type=int,
        default=5000,
        help="Max total prompt chars for cluster payload (default: 5000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    load_dependencies()

    LOGGER.info("Reading CSV: %s", args.input_csv)
    df = pd.read_csv(args.input_csv)

    text_col = detect_text_column(df, requested=args.text_col)
    if args.state_col not in df.columns:
        raise ValueError(f"State column '{args.state_col}' not found in input CSV.")
    if args.id_col not in df.columns:
        LOGGER.warning("ID column '%s' not found; continuing without explicit ID checks.", args.id_col)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Strip outcome language
    LOGGER.info("Stage 1: Stripping outcome language from text column '%s'", text_col)
    df["text_stripped"] = df[text_col].fillna("").astype(str).map(strip_outcome_language)

    # Preserve original text column for outputs, but use stripped text for embedding
    embedding_text_col = "text_stripped"

    summaries = []
    for state, grp in df.groupby(args.state_col, dropna=False):
        state_name = str(state) if pd.notna(state) else "UNKNOWN_STATE"
        summaries.append(
            process_state(
                state=state_name,
                df_state=grp.copy(),
                text_col=embedding_text_col,
                out_dir=args.output_dir,
                embedding_model=args.embedding_model,
                run_gpt=args.run_gpt_labeling,
                gpt_model=args.gpt_model,
                gpt_sample_cap=args.gpt_sample_cap,
                gpt_sample_char_cap=args.gpt_sample_char_cap,
                gpt_prompt_char_cap=args.gpt_prompt_char_cap,
                seed=args.seed,
            )
        )

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(args.output_dir / "all_states_summary.csv", index=False)
    LOGGER.info("Pipeline completed. Summary saved to %s", args.output_dir / "all_states_summary.csv")


if __name__ == "__main__":
    main()
