#!/usr/bin/env python3
import os
import sys
import glob
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


sys.path.insert(0, str(project_root()))
load_dotenv()


# ---- LLM summarization ----
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


def build_llm():
    summarize_model = os.getenv("SUMMARIZE_MODEL", "gpt-3.5-turbo")
    try:
        return ChatOpenAI(model=summarize_model, temperature=0)
    except TypeError:
        return ChatOpenAI(model_name=summarize_model, temperature=0)


LLM = build_llm()


def summarize(original_text: str) -> str:
    system = SystemMessage(content=(
        "You are an expert summarizer of sales call transcripts. "
        "Write a concise 1-2 sentence summary focusing on buyer needs, objections, and next steps."
    ))
    user = HumanMessage(content=original_text)
    try:
        msg = LLM.invoke([system, user])
        return getattr(msg, "content", str(msg))
    except Exception:
        out = LLM.invoke(original_text)
        return getattr(out, "content", str(out))


# ---- Continuous heuristics ----
import re
from rapidfuzz import fuzz


def token_overlap_ratio(a: str, b: str) -> float:
    at = set(t for t in a.lower().split() if t.isalnum())
    bt = set(t for t in b.lower().split() if t.isalnum())
    if not at:
        return 0.0
    return round(len(at & bt) / len(at), 3)


def rouge_like(a: str, b: str) -> float:
    return round(fuzz.token_set_ratio(a, b) / 100.0, 3)


def numeric_consistency(ctx: str, out: str) -> float:
    nums_ctx = set(re.findall(r"-?\d+(?:\.\d+)?", ctx))
    nums_out = set(re.findall(r"-?\d+(?:\.\d+)?", out))
    if not nums_out:
        return 0.0
    return round(len(nums_out & nums_ctx) / len(nums_out), 3)


def coherence_heuristic(summary: str) -> float:
    sents = [s.strip() for s in summary.split(".") if s.strip()]
    if not sents:
        return 0.0
    length = min(len(summary) / 400.0, 1.0)  # target roughly 1-2 concise sentences
    structure = min(len(sents) / 3.0, 1.0)
    return round(0.5 * length + 0.5 * structure, 3)


def compute_heuristics(original: str, summary: str) -> Dict[str, float]:
    return {
        "coherence_heuristic": coherence_heuristic(summary),
        "faithfulness_heuristic": rouge_like(summary, original),
        "completeness_heuristic": token_overlap_ratio(summary, original),
        "accuracy_heuristic": numeric_consistency(original, summary),
    }


def numeric_label_consistency(label: Any, numeric: Any) -> float:  # type: ignore[name-defined]
    # No labels in this dataset; return neutral confidence
    return 0.5


# ---- GEval metrics (model-based) ----
def build_geval_metrics():
    try:
        from deepeval.metrics import GEval
        from deepeval.models import OpenAIModel

        model = OpenAIModel(
            model=os.getenv("DEEPEVAL_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

        coherence = GEval(
            name="Coherence",
            criteria=(
                "Score 0.0-1.0 for: logical flow, internal consistency, clarity. "
                "Return ONLY a number."
            ),
            evaluation_params=["input", "actual_output"],
            model=model,
        )

        faithfulness = GEval(
            name="Faithfulness",
            criteria=(
                "Score 0.0-1.0 for grounding in provided context without hallucinations. "
                "Return ONLY a number."
            ),
            evaluation_params=["input", "actual_output", "retrieval_context"],
            model=model,
        )

        completeness = GEval(
            name="Completeness",
            criteria=(
                "Score 0.0-1.0 for covering key points implied by the prompt and context. "
                "Return ONLY a number."
            ),
            evaluation_params=["input", "actual_output", "retrieval_context"],
            model=model,
        )

        accuracy = GEval(
            name="Accuracy",
            criteria=(
                "Score 0.0-1.0 for factual correctness vs context (names, numbers, facts). "
                "Return ONLY a number."
            ),
            evaluation_params=["input", "actual_output", "retrieval_context"],
            model=model,
        )

        return coherence, faithfulness, completeness, accuracy
    except Exception:
        return None, None, None, None


# ---- IO helpers ----
def load_transcripts(transcripts_dir: Path) -> List[Tuple[str, str]]:
    paths = sorted(glob.glob(str(transcripts_dir / "*.txt")))
    out: List[Tuple[str, str]] = []
    for p in paths:
        text = Path(p).read_text(encoding="utf-8").strip()
        if text:
            out.append((os.path.basename(p), text))
    return out


def save_results_csv(results: List[Dict[str, Any]]) -> Path:
    out_dir = project_root() / "eval-results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"deepeval_results_{ts}.csv"
    import csv
    cols = [
        "id",
        # model-based
        "coherence_model",
        "faithfulness_model",
        "completeness_model",
        "accuracy_model",
        # heuristic
        "coherence_heuristic",
        "faithfulness_heuristic",
        "completeness_heuristic",
        "accuracy_heuristic",
        # rules
        "numeric_label_consistency",
    ]
    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in cols})
    print(f"Wrote {len(results)} rows to {out_file}")
    return out_file


def main() -> None:
    transcripts_dir = project_root() / "test-data" / "call_transcripts"
    items = load_transcripts(transcripts_dir)

    coherence, faithfulness, completeness, accuracy = build_geval_metrics()
    used_deepeval = coherence is not None

    results: List[Dict[str, Any]] = []
    total = len(items)
    for idx, (file_id, original) in enumerate(items, start=1):
        print(f"Processing {idx}/{total}: {file_id}")
        summary = summarize(original)

        # Heuristics against ground truth (original used as context)
        heur = compute_heuristics(original, summary)

        row: Dict[str, Any] = {
            "id": file_id,
            "coherence_model": 0.0,
            "faithfulness_model": 0.0,
            "completeness_model": 0.0,
            "accuracy_model": 0.0,
            **heur,
            "numeric_label_consistency": numeric_label_consistency(None, None),
        }

        # GEval model-based metrics (best effort)
        if used_deepeval:
            try:
                inp = "Summarize transcript focusing on needs, objections, next steps."
                ctx = original
                coh = coherence.measure(inp, summary, ctx)
                fai = faithfulness.measure(inp, summary, ctx)
                com = completeness.measure(inp, summary, ctx)
                acc = accuracy.measure(inp, summary, ctx)
                row.update({
                    "coherence_model": float(getattr(coh, "score", coh)),
                    "faithfulness_model": float(getattr(fai, "score", fai)),
                    "completeness_model": float(getattr(com, "score", com)),
                    "accuracy_model": float(getattr(acc, "score", acc)),
                })
            except Exception:
                pass

        results.append(row)

    save_results_csv(results)
    print("Completed DeepEval transcript evaluation" + (" (heuristic only)" if not used_deepeval else ""))


if __name__ == "__main__":
    main()


