"""
Evaluation Pipeline for Code Search & RAG Generation
Computes Precision@K, Recall@K, Hit@K, MRR, and Generation Quality against a Golden Dataset.
"""

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Import search & generation functions from search_cli
from search_cli import cmd_search, DEFAULT_DB_DIR, DEFAULT_COLLECTION

console = Console()
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


# -----------------------------------------------------------------------------
# 1. Metric Calculations
# -----------------------------------------------------------------------------
def compute_precision_at_k(retrieved_entities: List[str], target_entities: List[str], k: int) -> float:
    """
    Precision@K = (Number of relevant entities in top K) / K
    Measures how many of the top-K returned results were actually relevant.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_entities[:k]
    if not target_entities:
        # Negative test case: if no targets exist, precision is 1.0 if top_k is empty, else 0.0
        return 1.0 if len(top_k) == 0 else 0.0

    relevant_retrieved = sum(1 for item in top_k if item in target_entities)
    return relevant_retrieved / k


def compute_recall_at_k(retrieved_entities: List[str], target_entities: List[str], k: int) -> float:
    """
    Recall@K = (Number of relevant entities in top K) / (Total relevant entities)
    Measures the fraction of all relevant entities that were successfully retrieved in top K.
    """
    if not target_entities:
        # Negative test case: if nothing was supposed to match and none did, recall is 1.0
        return 1.0

    top_k = retrieved_entities[:k]
    relevant_retrieved = sum(1 for item in top_k if item in target_entities)
    return relevant_retrieved / len(target_entities)


def compute_hit_at_k(retrieved_entities: List[str], target_entities: List[str], k: int) -> float:
    """
    Hit@K = 1.0 if at least one relevant entity is in top K, else 0.0.
    """
    if not target_entities:
        return 1.0 if len(retrieved_entities[:k]) == 0 else 0.0
    top_k = retrieved_entities[:k]
    return 1.0 if any(item in target_entities for item in top_k) else 0.0


def compute_reciprocal_rank(retrieved_entities: List[str], target_entities: List[str]) -> float:
    """
    Reciprocal Rank = 1 / rank of first relevant entity (1-indexed). Returns 0.0 if not found.
    """
    if not target_entities:
        return 1.0 if len(retrieved_entities) == 0 else 0.0
    for idx, entity in enumerate(retrieved_entities, start=1):
        if entity in target_entities:
            return 1.0 / idx
    return 0.0


def evaluate_generation_keywords(generated_answer: str, expected_keywords: List[str]) -> float:
    """
    Measures keyword/fact recall in the generated answer text.
    """
    if not expected_keywords:
        return 1.0
    if not generated_answer:
        return 0.0

    lower_answer = generated_answer.lower()
    matches = sum(1 for kw in expected_keywords if kw.lower() in lower_answer)
    return matches / len(expected_keywords)


# -----------------------------------------------------------------------------
# 2. Main Evaluation Runner
# -----------------------------------------------------------------------------
def run_evaluation(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    db_path: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    eval_generation: bool = False,
    model: str = "llama3",
    k_values: List[int] = [1, 3, 5],
):
    if not dataset_path.exists():
        console.print(f"[red]Error: Dataset file '{dataset_path}' not found.[/red]")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    console.print(Panel(
        f"[bold cyan]Running Code Search & RAG Evaluation[/bold cyan]\n"
        f"Test Cases: [bold]{len(test_cases)}[/bold]  |  Evaluate Generation: [bold]{eval_generation}[/bold]  |  Model: [bold]{model}[/bold]",
        border_style="cyan"
    ))

    max_k = max(k_values)
    results = []

    for tc in test_cases:
        tc_id = tc["id"]
        query = tc["query"]
        target_entities = tc.get("target_entities", [])
        expected_keywords = tc.get("expected_answer_keywords", [])

        # Run search quietly by redirecting standard output
        sink = io.StringIO()
        try:
            with redirect_stdout(sink):
                retrieved_records, generated_text = cmd_search(
                    query=query,
                    limit=max_k,
                    db_path=db_path,
                    collection_name=collection_name,
                    generate=eval_generation,
                    model=model,
                )
        except Exception as e:
            console.print(f"[red]Failed query '{query}': {e}[/red]")
            retrieved_records, generated_text = [], ""

        retrieved_names = [r.get("name", "") for r in retrieved_records]

        # Calculate metrics
        row_metrics = {
            "id": tc_id,
            "query": query,
            "target": ", ".join(target_entities) if target_entities else "[none]",
            "retrieved_top1": retrieved_names[0] if retrieved_names else "[none]",
            "rr": compute_reciprocal_rank(retrieved_names, target_entities),
        }

        for k in k_values:
            row_metrics[f"p@{k}"] = compute_precision_at_k(retrieved_names, target_entities, k)
            row_metrics[f"r@{k}"] = compute_recall_at_k(retrieved_names, target_entities, k)
            row_metrics[f"hit@{k}"] = compute_hit_at_k(retrieved_names, target_entities, k)

        if eval_generation:
            row_metrics["gen_fact_recall"] = evaluate_generation_keywords(generated_text, expected_keywords)
            row_metrics["answer_preview"] = (generated_text[:90] + "...") if generated_text else "[no answer]"

        results.append(row_metrics)

    # -------------------------------------------------------------------------
    # 3. Render Detailed Results Table
    # -------------------------------------------------------------------------
    detail_table = Table(title="Evaluation Per Test Query", header_style="bold magenta", border_style="dim")
    detail_table.add_column("ID", style="bold cyan", width=8)
    detail_table.add_column("Query", width=30)
    detail_table.add_column("Target Entity", style="yellow", width=18)
    detail_table.add_column("Top-1 Match", width=18)
    detail_table.add_column("P@1", justify="center")
    detail_table.add_column("P@3", justify="center")
    detail_table.add_column("R@3", justify="center")
    detail_table.add_column("Hit@3", justify="center")
    detail_table.add_column("MRR", justify="center")

    if eval_generation:
        detail_table.add_column("Gen Fact Recall", justify="center", style="bold green")

    for r in results:
        cols = [
            r["id"],
            r["query"],
            r["target"],
            r["retrieved_top1"],
            f"{r['p@1']:.2f}",
            f"{r['p@3']:.2f}",
            f"{r['r@3']:.2f}",
            f"{r['hit@3']:.1f}",
            f"{r['rr']:.2f}",
        ]
        if eval_generation:
            cols.append(f"{r['gen_fact_recall']:.2f}")
        detail_table.add_row(*cols)

    console.print(detail_table)
    console.print()

    # -------------------------------------------------------------------------
    # 4. Render Summary Scorecard
    # -------------------------------------------------------------------------
    n = len(results)
    avg_p1 = sum(r["p@1"] for r in results) / n
    avg_p3 = sum(r["p@3"] for r in results) / n
    avg_p5 = sum(r.get("p@5", 0.0) for r in results) / n

    avg_r1 = sum(r["r@1"] for r in results) / n
    avg_r3 = sum(r["r@3"] for r in results) / n
    avg_r5 = sum(r.get("r@5", 0.0) for r in results) / n

    avg_hit1 = sum(r["hit@1"] for r in results) / n
    avg_hit3 = sum(r["hit@3"] for r in results) / n
    avg_hit5 = sum(r.get("hit@5", 0.0) for r in results) / n

    mrr = sum(r["rr"] for r in results) / n

    summary_table = Table(title="Aggregate Retrieval Performance Scorecard", header_style="bold green", border_style="cyan")
    summary_table.add_column("Metric", style="bold white")
    summary_table.add_column("Score (@1)", justify="center")
    summary_table.add_column("Score (@3)", justify="center")
    summary_table.add_column("Score (@5)", justify="center")

    summary_table.add_row("Precision@K (Fraction of retrieved that are relevant)", f"{avg_p1:.4f}", f"{avg_p3:.4f}", f"{avg_p5:.4f}")
    summary_table.add_row("Recall@K (Fraction of all relevant items retrieved)", f"{avg_r1:.4f}", f"{avg_r3:.4f}", f"{avg_r5:.4f}")
    summary_table.add_row("Hit@K (Success rate: at least 1 relevant found)", f"{avg_hit1:.2%}", f"{avg_hit3:.2%}", f"{avg_hit5:.2%}")
    summary_table.add_row("MRR (Mean Reciprocal Rank)", f"{mrr:.4f}", "-", "-")

    console.print(summary_table)

    if eval_generation:
        avg_gen_recall = sum(r["gen_fact_recall"] for r in results) / n
        console.print(f"\n[bold green]Mean Generation Fact Recall:[/bold green] [bold cyan]{avg_gen_recall:.2%}[/bold cyan]")


# -----------------------------------------------------------------------------
# 3. CLI Arguments
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate Code Search & Generation Pipeline")
    parser.add_argument(
        "--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to golden dataset JSON"
    )
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_DIR), help="Path to local Qdrant database folder"
    )
    parser.add_argument(
        "--collection", type=str, default=DEFAULT_COLLECTION, help="Name of Qdrant collection"
    )
    parser.add_argument(
        "--eval-generation", "-g", action="store_true", help="Also evaluate LLM generated answers with Ollama"
    )
    parser.add_argument(
        "--model", "-m", type=str, default="llama3", help="Ollama model to evaluate (default: llama3)"
    )

    args = parser.parse_args()
    run_evaluation(
        dataset_path=Path(args.dataset),
        db_path=Path(args.db),
        collection_name=args.collection,
        eval_generation=args.eval_generation,
        model=args.model,
    )


if __name__ == "__main__":
    main()
