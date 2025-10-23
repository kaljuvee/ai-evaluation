import os
import csv
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()


def build_llm():
    try:
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    except TypeError:
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


llm = build_llm()


def numeric_label_consistency(label: Any, numeric: Any) -> float:  # type: ignore[name-defined]
    mapping = {
        "very high": 5,
        "very strong": 5,
        "high": 4,
        "strong": 4,
        "moderate": 3,
        "mixed/watchful": 3,
        "low": 2,
        "very low": 1,
    }
    try:
        if isinstance(numeric, str) and numeric.isdigit():
            numeric = int(numeric)
        elif isinstance(numeric, (float, int)):
            numeric = int(round(float(numeric)))
        else:
            numeric = None
    except Exception:
        numeric = None

    if isinstance(label, str):
        key = label.strip().lower()
        expected = mapping.get(key)
        if expected and numeric:
            return 1.0 if expected == numeric else 0.0
    return 0.5


def summarize_text(source_text: str) -> str:
    prompt = (
        "Summarize the following sales call transcript in one to two sentences. "
        "Focus on buyer needs, objections, and next steps.\n\n" + source_text
    )
    try:
        return llm.predict(prompt)  # Older langchain interface
    except Exception:
        message = llm.invoke(prompt)  # Newer langchain interface
        return getattr(message, "content", str(message))


def compute_heuristics(inp: str, out: str, ctx: str) -> Dict[str, float]:
    out_clean = (out or "").strip()
    coherence_h = 1.0 if len(out_clean.split()) > 12 and ("." in out_clean or ";" in out_clean) else 0.6
    faithfulness_h = 1.0 if ctx and any(k in out_clean.lower() for k in ["%", "usd", "uae", "interest"]) else 0.6
    atok = set([t for t in out_clean.lower().split() if t.isalpha() or t.isalnum()])
    ctok = set([t for t in (ctx or "").lower().split() if t.isalpha() or t.isalnum()])
    completeness_h = round(len(atok & ctok) / max(1, len(atok)), 3) if atok else 0.0
    accuracy_h = 1.0 if any(ch.isdigit() for ch in out_clean) and any(ch.isdigit() for ch in ctx) else 0.6
    return {
        "coherence_heuristic": round(coherence_h, 3),
        "faithfulness_heuristic": round(faithfulness_h, 3),
        "completeness_heuristic": completeness_h,
        "accuracy_heuristic": round(accuracy_h, 3),
    }


def save_results_csv(results: List[Dict[str, Any]]) -> Path:  # type: ignore[name-defined]
    out_dir = Path("eval-results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"deepeval_results_{ts}.csv"
    cols = [
        "id",
        "coherence_model",
        "faithfulness_model",
        "completeness_model",
        "accuracy_model",
        "coherence_heuristic",
        "faithfulness_heuristic",
        "completeness_heuristic",
        "accuracy_heuristic",
        "numeric_label_consistency",
    ]
    with out_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in cols})
    print(f"Wrote {len(results)} rows to {out_file}")
    return out_file


def run_deepeval_over_transcripts(
    transcripts_dir: str = "test-data/call_transcripts",
) -> Path:  # type: ignore[name-defined]
    transcript_paths = sorted(glob.glob(os.path.join(transcripts_dir, "*.txt")))
    results: List[Dict[str, Any]] = []  # type: ignore[name-defined]
    used_deepeval = False

    # Prepare GEval metrics (best effort)
    try:
        from deepeval.metrics import GEval
        from deepeval.models import OpenAIModel

        model = OpenAIModel(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
        )

        coherence = GEval(
            name="Coherence",
            evaluation_params=[
                "Is the response logically structured?",
                "Are claims internally consistent?",
            ],
            model=model,
        )

        faithfulness = GEval(
            name="Faithfulness",
            evaluation_params=[
                "Does the response rely on provided evidence?",
                "Avoids introducing unsupported facts?",
            ],
            model=model,
        )

        completeness = GEval(
            name="Completeness",
            evaluation_params=[
                "Does the response cover key aspects implied by the prompt?",
                "Does it reflect salient points from the evidence?",
            ],
            model=model,
        )

        accuracy = GEval(
            name="Accuracy",
            evaluation_params=[
                "Are specific facts (numbers, dates, names) consistent with evidence?",
                "Avoids factual errors and misinterpretations?",
            ],
            model=model,
        )
    except Exception as e:
        coherence = faithfulness = completeness = accuracy = None  # type: ignore[assignment]

    for tp in transcript_paths:
        text = Path(tp).read_text(encoding="utf-8").strip()
        if not text:
            continue
        summary = summarize_text(text)
        inp = "Summarize transcript"
        ctx = text

        heur = compute_heuristics(inp, summary, ctx)

        row: Dict[str, Any] = {
            "id": os.path.basename(tp),
            "coherence_model": 0,
            "faithfulness_model": 0,
            "completeness_model": 0,
            "accuracy_model": 0,
            **heur,
            "numeric_label_consistency": numeric_label_consistency(None, None),
        }

        # Try DeepEval model-based scoring
        try:
            if coherence is not None:
                coh = coherence.measure(inp, summary, ctx)
                fai = faithfulness.measure(inp, summary, ctx)
                com = completeness.measure(inp, summary, ctx)
                acc = accuracy.measure(inp, summary, ctx)
                row.update(
                    {
                        "coherence_model": coh if isinstance(coh, (int, float)) else getattr(coh, "score", 0),
                        "faithfulness_model": fai if isinstance(fai, (int, float)) else getattr(fai, "score", 0),
                        "completeness_model": com if isinstance(com, (int, float)) else getattr(com, "score", 0),
                        "accuracy_model": acc if isinstance(acc, (int, float)) else getattr(acc, "score", 0),
                    }
                )
                used_deepeval = True
        except Exception:
            pass

        results.append(row)

    out_path = save_results_csv(results)
    print("Completed DeepEval evaluation" + (" (heuristic fallback)" if not used_deepeval else ""))
    return out_path


if __name__ == "__main__":
    run_deepeval_over_transcripts()

