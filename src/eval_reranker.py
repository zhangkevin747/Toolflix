"""
Evaluate the reranker against baselines using existing feedback.

For each feedback record we know:
  - The retriever candidates (with similarity scores)
  - Which tool the agent selected
  - Whether it succeeded (ground truth)

We train the reranker on a train split, then on the test split we ask:
  "If the agent always picked rank 1, how often would it succeed?"

Baselines:
  1. Retriever only (cosine similarity)
  2. BM-25 (lexical)
  3. Random
  4. Success-rate prior (wide-only proxy)

Ablations:
  5. Reranker (wide + deep)
  6. Wide-only (zero deep features)
  7. Deep-only (zero wide features)
"""
import json
import random
import sys
import numpy as np
from collections import defaultdict
from pathlib import Path
from rank_bm25 import BM25Okapi

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from reranker import Reranker


def load_feedback(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ("retriever_candidates" in r and "selected" in r
                    and "rating" in r and "error" not in r):
                records.append(r)
    return records


def retriever_rank1_key(record: dict) -> str:
    """Return the key of the retriever's #1 candidate."""
    c = record["retriever_candidates"][0]
    return f"{c['server_id']}/{c['tool_name']}"


def success_rate_ranking(records_for_prior: list[dict], record: dict) -> list[dict]:
    """Rank candidates by historical success rate (oracle-ish baseline)."""
    tool_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in records_for_prior:
        sel = r["selected"]
        key = f"{sel['server_id']}/{sel['tool_name']}"
        tool_stats[key]["total"] += 1
        if r["rating"].get("success"):
            tool_stats[key]["success"] += 1

    candidates = record["retriever_candidates"]
    scored = []
    for c in candidates:
        key = f"{c['server_id']}/{c['tool_name']}"
        stats = tool_stats[key]
        rate = stats["success"] / stats["total"] if stats["total"] > 0 else 0.5
        scored.append({**c, "score": rate})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def compute_auc_and_pairwise(test_records: list[dict], rank_fn):
    """Compute AUC and pairwise accuracy."""
    scores_pos = []
    scores_neg = []
    pair_correct = 0
    pair_total = 0

    for r in test_records:
        ranked = rank_fn(r)
        if not ranked:
            continue

        sel_key = f"{r['selected']['server_id']}/{r['selected']['tool_name']}"
        success = r["rating"].get("success", False)

        candidate_scores = {}
        for c in ranked:
            key = f"{c['server_id']}/{c['tool_name']}"
            candidate_scores[key] = c.get("rerank_score", c.get("score", c.get("similarity", 0)))

        sel_score = candidate_scores.get(sel_key)
        if sel_score is None:
            continue

        if success:
            scores_pos.append(sel_score)
        else:
            scores_neg.append(sel_score)

        other_scores = [s for k, s in candidate_scores.items() if k != sel_key]
        for other_s in other_scores:
            pair_total += 1
            if success and sel_score > other_s:
                pair_correct += 1
            elif not success and sel_score < other_s:
                pair_correct += 1
            elif sel_score == other_s:
                pair_correct += 0.5

    auc_correct = 0
    auc_total = 0
    for sp in scores_pos:
        for sn in scores_neg:
            auc_total += 1
            if sp > sn:
                auc_correct += 1
            elif sp == sn:
                auc_correct += 0.5

    auc = auc_correct / auc_total if auc_total else 0.0
    pairwise_acc = pair_correct / pair_total if pair_total else 0.0

    return auc, pairwise_acc, len(scores_pos), len(scores_neg)


def evaluate_ranking(test_records: list[dict], rank_fn, top_k: int = 5, label: str = ""):
    """Evaluate a ranking function."""
    selected_in_top1 = 0
    selected_in_topk = 0
    correct_direction = 0
    cat_stats = defaultdict(lambda: {"total": 0, "selected_top1": 0, "selected_topk": 0})
    success_ranks = []
    fail_ranks = []

    for r in test_records:
        ranked = rank_fn(r)
        if not ranked:
            continue

        sel_key = f"{r['selected']['server_id']}/{r['selected']['tool_name']}"
        success = r["rating"].get("success", False)
        category = r["category"]

        sel_rank = None
        for i, c in enumerate(ranked):
            key = f"{c['server_id']}/{c['tool_name']}"
            if key == sel_key:
                sel_rank = i + 1
                break

        if sel_rank is None:
            continue

        if sel_rank == 1:
            selected_in_top1 += 1
            cat_stats[category]["selected_top1"] += 1
        if sel_rank <= top_k:
            selected_in_topk += 1
            cat_stats[category]["selected_topk"] += 1
        cat_stats[category]["total"] += 1

        if success:
            success_ranks.append(sel_rank)
        else:
            fail_ranks.append(sel_rank)

        if success and sel_rank <= top_k:
            correct_direction += 1
        elif not success and sel_rank > top_k:
            correct_direction += 1

    total = sum(cs["total"] for cs in cat_stats.values())

    auc, pairwise_acc, n_pos, n_neg = compute_auc_and_pairwise(test_records, rank_fn)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if total == 0:
        print("  No records matched — skipping metrics.")
        return
    print(f"  Selected tool at rank 1: {selected_in_top1}/{total} ({100*selected_in_top1/total:.1f}%)")
    print(f"  Selected tool in top {top_k}:  {selected_in_topk}/{total} ({100*selected_in_topk/total:.1f}%)")
    print(f"  Correct direction:       {correct_direction}/{total} ({100*correct_direction/total:.1f}%)")
    if success_ranks:
        print(f"  Avg rank of succeeded:   {np.mean(success_ranks):.2f}  (lower=better, ideal=1)")
    if fail_ranks:
        print(f"  Avg rank of failed:      {np.mean(fail_ranks):.2f}  (higher=better)")
    print(f"  AUC:                     {auc:.4f}  ({n_pos} pos, {n_neg} neg)")
    print(f"  Pairwise accuracy:       {pairwise_acc:.4f}")

    print(f"\n  Per-category (selected in top {top_k}):")
    for cat in ["fetch", "pdf", "search", "filesystem"]:
        cs = cat_stats[cat]
        if cs["total"] > 0:
            print(f"    {cat:12s}: {cs['selected_topk']}/{cs['total']} ({100*cs['selected_topk']/cs['total']:.1f}%)")

    return {
        "top1": selected_in_top1 / total if total else 0,
        "topk": selected_in_topk / total if total else 0,
        "correct_dir": correct_direction / total if total else 0,
        "avg_success_rank": np.mean(success_ranks) if success_ranks else 0,
        "avg_fail_rank": np.mean(fail_ranks) if fail_ranks else 0,
        "auc": auc,
        "pairwise_acc": pairwise_acc,
    }


def build_dense_retriever_ranking_fn(model_name: str, tool_texts: dict[str, str],
                                     use_automodel: bool = False):
    """Build a ranking function from any dense encoder.

    Args:
        model_name: HuggingFace model ID.
        tool_texts: {endpoint_key: description_text} for all tools.
        use_automodel: If True, use AutoModel + CLS pooling + L2 norm
                       (needed for ToolRet BGE which lacks SentenceTransformer packaging).
                       If False, use SentenceTransformer.
    """
    import torch

    keys = list(tool_texts.keys())
    texts = [tool_texts[k] for k in keys]
    key_to_idx = {k: i for i, k in enumerate(keys)}

    if use_automodel:
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        def _encode_batch(batch_texts, batch_size=64):
            all_embs = []
            for i in range(0, len(batch_texts), batch_size):
                chunk = batch_texts[i:i + batch_size]
                inputs = tokenizer(chunk, padding=True, truncation=True,
                                   max_length=512, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                # CLS pooling + L2 normalize
                embs = outputs.last_hidden_state[:, 0]
                embs = F.normalize(embs, p=2, dim=1)
                all_embs.append(embs.cpu().numpy())
            return np.vstack(all_embs)

        print(f"  Encoding {len(texts)} tools with {model_name} (AutoModel)...")
        tool_embs = _encode_batch(texts)
        tool_norms = np.linalg.norm(tool_embs, axis=1)

        # Cache query embeddings
        _query_cache = {}

        def ranking_fn(record):
            query = record.get("query", record.get("task", ""))
            if query not in _query_cache:
                _query_cache[query] = _encode_batch([query])[0]
            q_emb = _query_cache[query]
            q_norm = np.linalg.norm(q_emb)

            candidates = record["retriever_candidates"]
            scored = []
            for c in candidates:
                key = f"{c['server_id']}/{c['tool_name']}"
                idx = key_to_idx.get(key)
                if idx is not None:
                    sim = float(np.dot(tool_embs[idx], q_emb) / (tool_norms[idx] * q_norm + 1e-9))
                else:
                    sim = 0.0
                scored.append({**c, "rerank_score": sim})
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored

    else:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(model_name)
        print(f"  Encoding {len(texts)} tools with {model_name} (SentenceTransformer)...")
        tool_embs = st_model.encode(texts, show_progress_bar=False)
        tool_norms = np.linalg.norm(tool_embs, axis=1)

        _query_cache = {}

        def ranking_fn(record):
            query = record.get("query", record.get("task", ""))
            if query not in _query_cache:
                _query_cache[query] = st_model.encode([query])[0]
            q_emb = _query_cache[query]
            q_norm = np.linalg.norm(q_emb)

            candidates = record["retriever_candidates"]
            scored = []
            for c in candidates:
                key = f"{c['server_id']}/{c['tool_name']}"
                idx = key_to_idx.get(key)
                if idx is not None:
                    sim = float(np.dot(tool_embs[idx], q_emb) / (tool_norms[idx] * q_norm + 1e-9))
                else:
                    sim = 0.0
                scored.append({**c, "rerank_score": sim})
            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored

    return ranking_fn


# External baseline model configs: (label, hf_model_id, use_automodel)
EXTERNAL_BASELINES = [
    # ToolRet (ACL 2025) — fine-tuned BGE-large on 43K tool retrieval data
    ("ToolRet BGE-large (fine-tuned)",
     "mangopy/ToolRet-trained-bge-large-en-v1.5", True),
    # COLT Phase-1 (CIKM 2024) — fine-tuned Contriever on ToolBench
    ("COLT Contriever (fine-tuned)",
     "Tool-COLT/contriever-base-msmarco-v1-ToolBenchG3", False),
    # General-purpose encoders for multi-encoder comparison
    ("BGE-base-en-v1.5 (general-purpose)",
     "BAAI/bge-base-en-v1.5", False),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-baselines", action="store_true",
                        help="Include external baselines (ToolRet, COLT) — requires downloading models")
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    feedback_path = str(data_dir / "feedback.jsonl")
    embeddings_path = str(data_dir / "embeddings.json")

    print("Loading feedback...")
    all_records = load_feedback(feedback_path)
    print(f"Loaded {len(all_records)} records")

    # Train/test split: 80/20 by time (first 80% for training, last 20% for eval)
    split = int(len(all_records) * 0.8)
    train_records = all_records[:split]
    test_records = all_records[split:]
    print(f"Train: {len(train_records)}, Test: {len(test_records)}")

    # Write a temporary train-only feedback file for the reranker
    train_feedback_path = str(data_dir / "_train_feedback.jsonl")
    with open(train_feedback_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    # --- Baseline 1: Retriever only ---
    def retriever_ranking(record):
        return sorted(record["retriever_candidates"],
                      key=lambda c: c.get("similarity", 0), reverse=True)

    evaluate_ranking(test_records, retriever_ranking, label="Baseline: Retriever (cosine similarity)")

    # --- Baseline 2: BM-25 ---
    # Build BM25 index over all tool descriptions from embeddings.json
    with open(embeddings_path) as f:
        all_endpoints = json.load(f)
    ep_key_to_text = {}
    for ep in all_endpoints:
        key = f"{ep['server_id']}/{ep['tool_name']}"
        ep_key_to_text[key] = ep.get("text", ep.get("tool_name", ""))

    # Build index over all endpoints
    all_keys = list(ep_key_to_text.keys())
    all_texts_tokenized = [ep_key_to_text[k].lower().split() for k in all_keys]
    bm25_index = BM25Okapi(all_texts_tokenized)
    bm25_key_to_idx = {k: i for i, k in enumerate(all_keys)}

    def bm25_ranking(record):
        query = record.get("query", record.get("task", ""))
        query_tokens = query.lower().split()
        scores = bm25_index.get_scores(query_tokens)
        candidates = record["retriever_candidates"]
        scored = []
        for c in candidates:
            key = f"{c['server_id']}/{c['tool_name']}"
            idx = bm25_key_to_idx.get(key)
            bm25_score = scores[idx] if idx is not None else 0.0
            scored.append({**c, "rerank_score": bm25_score})
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored

    evaluate_ranking(test_records, bm25_ranking, label="Baseline: BM-25 (lexical)")

    # --- Baseline 3: Random ---
    def random_ranking(record):
        cands = list(record["retriever_candidates"])
        random.shuffle(cands)
        return cands

    random.seed(42)
    evaluate_ranking(test_records, random_ranking, label="Baseline: Random")

    # --- Baseline 4: Success-rate prior ---
    def success_prior_ranking(record):
        return success_rate_ranking(train_records, record)

    evaluate_ranking(test_records, success_prior_ranking, label="Baseline: Success-rate prior (wide-only proxy)")

    # --- External baselines (ToolRet, COLT, multi-encoder) ---
    if args.external_baselines:
        # ep_key_to_text already built for BM-25 above
        for label, model_id, use_auto in EXTERNAL_BASELINES:
            print(f"\nLoading external baseline: {label}...")
            try:
                ext_fn = build_dense_retriever_ranking_fn(
                    model_id, ep_key_to_text, use_automodel=use_auto)
                evaluate_ranking(test_records, ext_fn, label=f"External: {label}")
            except Exception as e:
                print(f"  SKIPPED {label}: {e}")

    # --- Reranker: Wide + Deep ---
    print("\n\nTraining reranker on train split...")
    reranker = Reranker(
        embeddings_path=embeddings_path,
        feedback_path=train_feedback_path,
        model_path=None,  # Train fresh
    )
    reranker.train_on_feedback(train_feedback_path, epochs=50)

    def _make_ranking_fn(zero_wide=False, zero_deep=False):
        """Create a ranking function with optional feature ablation."""
        import torch

        def ranking_fn(record):
            query = record.get("query", record.get("task", ""))
            candidates = record["retriever_candidates"]
            enriched = list(candidates)

            if not zero_wide and not zero_deep:
                # Normal reranker
                return reranker.rerank(enriched, query, top_k=len(enriched), explore=False)

            # Ablation: manually score with zeroed features
            query_emb = reranker._safe_encode([query])[0]
            query_tensor = torch.tensor(query_emb, dtype=torch.float32).unsqueeze(0)
            model_idx_tensor = torch.tensor([reranker._model_idx], dtype=torch.long)

            scored = []
            for c in enriched:
                key = f"{c['server_id']}/{c['tool_name']}"
                idx = reranker.endpoint_to_idx.get(key)
                if idx is None:
                    scored.append({**c, "rerank_score": c.get("similarity", 0)})
                    continue

                wide_feat = reranker.wide.get_features(key, reranker.model_name, c.get("similarity", 0.0))
                wide_tensor = torch.tensor([wide_feat], dtype=torch.float32)
                tool_tensor = reranker.endpoint_desc_embs[idx].unsqueeze(0)
                idx_tensor = torch.tensor([idx], dtype=torch.long)

                if zero_wide:
                    wide_tensor = torch.zeros_like(wide_tensor)

                with torch.no_grad():
                    if zero_deep:
                        zero_query = torch.zeros_like(query_tensor)
                        zero_tool = torch.zeros_like(tool_tensor)
                        score = reranker.model(wide_tensor, model_idx_tensor, zero_query, zero_tool, idx_tensor)
                    else:
                        score = reranker.model(wide_tensor, model_idx_tensor, query_tensor, tool_tensor, idx_tensor)

                scored.append({**c, "rerank_score": score.item()})

            scored.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored

        return ranking_fn

    evaluate_ranking(test_records, _make_ranking_fn(),
                     label="Reranker: Wide + Deep (full model)")
    evaluate_ranking(test_records, _make_ranking_fn(zero_wide=True),
                     label="Ablation: Deep only (zero wide features)")
    evaluate_ranking(test_records, _make_ranking_fn(zero_deep=True),
                     label="Ablation: Wide only (zero deep features)")

    # Cleanup
    Path(train_feedback_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
