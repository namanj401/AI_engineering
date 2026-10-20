"""
Code Search CLI (search_cli.py)
Hybrid vector search (Dense + Sparse) with AST code parsing, Qdrant on-disk storage,
and Cross-Encoder reranking.
"""

import argparse
import ast
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Core Libraries
import ollama
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from sentence_transformers import CrossEncoder

# Configuration Defaults
DEFAULT_COLLECTION = "code_search"
DEFAULT_DB_DIR = Path(__file__).resolve().parent / "local_qdrant_db"
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DENSE_DIM = 384

EXCLUDE_DIRS = {".venv", ".git", "__pycache__", "node_modules", "dist", "build", ".pytest_cache"}
EXCLUDE_FILES = {"__init__.py", ".DS_Store"}

console = Console()


# -----------------------------------------------------------------------------
# 1. Directory Traversal and AST Parsing
# -----------------------------------------------------------------------------
def extract_chunks_from_file(file_path: Path) -> List[Dict[str, Any]]:
    """Reads a Python file and uses AST to extract functions and classes."""
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        console.print(f"[yellow]Skipping {file_path.name}: {e}[/yellow]")
        return []

    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError:
        return []

    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            code_segment = ast.get_source_segment(content, node)
            if not code_segment:
                continue

            entity_type = "ClassDef" if isinstance(node, ast.ClassDef) else "FunctionDef"
            docstring = ast.get_docstring(node) or ""

            chunks.append({
                "name": node.name,
                "type": entity_type,
                "file_path": str(file_path.resolve()),
                "start_line": node.lineno,
                "end_line": getattr(node, "end_lineno", node.lineno),
                "docstring": docstring,
                "code": code_segment,
            })

    return chunks


def scan_directory(target_dir: Path) -> List[Dict[str, Any]]:
    """Traverses a directory with pruning and extracts all code chunks."""
    all_chunks = []
    scanned_files = 0

    for dirpath, dirnames, filenames in target_dir.walk(top_down=True):
        # In-place directory pruning
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]

        for filename in filenames:
            if filename in EXCLUDE_FILES or filename.startswith("."):
                continue

            if filename.endswith(".py"):
                file_path = dirpath / filename
                scanned_files += 1
                chunks = extract_chunks_from_file(file_path)
                all_chunks.extend(chunks)

    console.print(f"[green]Scanned [bold]{scanned_files}[/bold] Python files, extracted [bold]{len(all_chunks)}[/bold] code chunks.[/green]")
    return all_chunks


# -----------------------------------------------------------------------------
# 2. Qdrant Client & Collection Setup
# -----------------------------------------------------------------------------
def get_qdrant_client(db_path: Path) -> QdrantClient:
    """Initializes local on-disk Qdrant client."""
    db_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(db_path))


def ensure_collection(client: QdrantClient, collection_name: str):
    """Creates hybrid collection if it doesn't already exist."""
    collections = [col.name for col in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams()
            },
        )
        console.print(f"[blue]Created collection '[bold]{collection_name}[/bold]' with Dense + Sparse support.[/blue]")


# -----------------------------------------------------------------------------
# 3. Index Command
# -----------------------------------------------------------------------------
def cmd_index(directory_path: str, db_path: Path = DEFAULT_DB_DIR, collection_name: str = DEFAULT_COLLECTION):
    """Indexes all code chunks in the given directory into Qdrant."""
    target_dir = Path(directory_path).resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        console.print(f"[red]Error: Path '{directory_path}' is not a valid directory.[/red]")
        sys.exit(1)

    console.print(f"[bold cyan]Traversing directory:[/bold cyan] {target_dir}")
    chunks = scan_directory(target_dir)
    if not chunks:
        console.print("[yellow]No code chunks found to index.[/yellow]")
        return

    console.print("[cyan]Loading embedding models (Dense: BGE-small, Sparse: BM25)...[/cyan]")
    dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    client = get_qdrant_client(db_path)
    ensure_collection(client, collection_name)

    # Texts to embed (name + docstring + code)
    texts = [
        f"{chunk['name']} ({chunk['type']})\n{chunk['docstring']}\n{chunk['code']}"
        for chunk in chunks
    ]

    console.print(f"[cyan]Generating dense and sparse embeddings for {len(chunks)} chunks...[/cyan]")
    dense_embeddings = list(dense_model.embed(texts))
    sparse_embeddings = list(sparse_model.embed(texts))

    points = []
    for i, chunk in enumerate(chunks):
        # Deterministic UUID5 for idempotent updates
        unique_key = f"{chunk['file_path']}:{chunk['name']}:{chunk['start_line']}"
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, unique_key))

        dense_vec = dense_embeddings[i].tolist()
        sparse_vec = models.SparseVector(
            indices=sparse_embeddings[i].indices.tolist(),
            values=sparse_embeddings[i].values.tolist(),
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec,
                },
                payload=chunk,
            )
        )

    console.print(f"[cyan]Upserting {len(points)} points into Qdrant...[/cyan]")
    # Batch upsert
    batch_size = 64
    for start_idx in range(0, len(points), batch_size):
        batch = points[start_idx : start_idx + batch_size]
        client.upsert(collection_name=collection_name, points=batch)

    console.print(f"[bold green]Successfully indexed {len(points)} code chunks into '{collection_name}'![/bold green]")


# -----------------------------------------------------------------------------
# 4. Search & Re-rank Command (+ Optional LLM Generation)
# -----------------------------------------------------------------------------
def generate_answer(query: str, contexts: List[Dict[str, Any]], model: str = "llama3"):
    """Formats retrieved code snippets and generates an answer using local Ollama."""
    console.print(f"\n[bold magenta]🤖 Generating Answer with {model}...[/bold magenta]\n")

    context_text = "\n\n".join([
        f"### File: {item.get('file_path', 'unknown')} (lines {item.get('start_line', '?')}-{item.get('end_line', '?')})\n"
        f"Entity: {item.get('name', 'unknown')} ({item.get('type', 'Code')})\n"
        f"```python\n{item.get('code', '')}\n```"
        for item in contexts
    ])

    system_prompt = (
        "You are an expert software engineer assistant. Answer the user's question "
        "using ONLY the provided code snippets from the codebase. "
        "Reference the file names and function/class names where appropriate. "
        "If the answer cannot be determined from the code, state that clearly."
    )

    user_prompt = f"Context from Codebase:\n{context_text}\n\nUser Question: {query}"

    try:
        response_stream = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )

        full_response = ""
        for chunk in response_stream:
            content = chunk["message"]["content"]
            full_response += content
            console.print(content, end="")
        console.print("\n")
        return full_response

    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")
        console.print("[dim]Make sure Ollama is installed and running (`ollama serve`).[/dim]")
        return ""


def cmd_search(
    query: str,
    limit: int = 5,
    db_path: Path = DEFAULT_DB_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    generate: bool = False,
    model: str = "llama3",
):
    """Executes hybrid prefetch query, re-ranks using CrossEncoder, and displays results."""
    client = get_qdrant_client(db_path)

    # Check collection existence
    collections = [col.name for col in client.get_collections().collections]
    if collection_name not in collections:
        console.print(f"[red]Error: Collection '{collection_name}' does not exist. Run 'index' first.[/red]")
        sys.exit(1)

    console.print(f"[cyan]Loading models for query: '[bold]{query}[/bold]'...[/cyan]")
    dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    # Embed query
    query_dense = list(dense_model.embed([query]))[0].tolist()
    query_sparse_raw = list(sparse_model.embed([query]))[0]
    query_sparse = models.SparseVector(
        indices=query_sparse_raw.indices.tolist(),
        values=query_sparse_raw.values.tolist(),
    )

    # 1. Hybrid search with RRF prefetch (retrieve candidate pool of candidates)
    prefetch_limit = max(limit * 4, 20)
    response = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_dense,
                using="dense",
                limit=prefetch_limit,
            ),
            models.Prefetch(
                query=query_sparse,
                using="sparse",
                limit=prefetch_limit,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=prefetch_limit,
    )

    candidate_points = response.points
    if not candidate_points:
        console.print("[yellow]No matching code snippets found.[/yellow]")
        return [], ""

    # 2. Re-rank candidates using Cross-Encoder
    candidate_payloads = [p.payload for p in candidate_points]
    pairs = [[query, p.get("code", "")] for p in candidate_payloads]
    scores = reranker.predict(pairs)

    # Attach re-rank scores and sort
    ranked_results = []
    for payload, score in zip(candidate_payloads, scores):
        ranked_results.append(payload | {"rerank_score": float(score)})

    ranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
    top_results = ranked_results[:limit]

    # 3. Pretty Terminal Display
    console.print(f"\n[bold green]Top {len(top_results)} Results for:[/bold green] [bold cyan]\"{query}\"[/bold cyan]\n")

    for rank, item in enumerate(top_results, start=1):
        name = item.get("name", "Unknown")
        entity_type = item.get("type", "Code")
        file_path = item.get("file_path", "")
        start_line = item.get("start_line", "?")
        end_line = item.get("end_line", "?")
        score = item.get("rerank_score", 0.0)
        code = item.get("code", "")

        # Syntax highlighted code block
        syntax = Syntax(
            code,
            "python",
            theme="monokai",
            line_numbers=True,
            start_line=start_line if isinstance(start_line, int) else 1,
        )

        title = f"[bold white]#{rank} [{entity_type}][/bold white] [bold yellow]{name}[/bold yellow]  |  Re-rank Score: [bold green]{score:.4f}[/bold green]"
        subtitle = f"[dim]{file_path}:{start_line}-{end_line}[/dim]"

        panel = Panel(
            syntax,
            title=title,
            subtitle=subtitle,
            title_align="left",
            subtitle_align="right",
            border_style="cyan",
            padding=(1, 2),
        )
        console.print(panel)
        console.print()

    # 4. LLM Generation step (if requested)
    generated_text = ""
    if generate:
        generated_text = generate_answer(query, top_results, model=model)

    return top_results, generated_text


# -----------------------------------------------------------------------------
# 5. CLI Entrypoint (argparse)
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Code Search CLI: Hybrid dense/sparse indexing and cross-encoder search for Python codebases."
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # Command: index
    index_parser = subparsers.add_parser("index", help="Index a directory of Python files into Qdrant")
    index_parser.add_argument("path", type=str, help="Path to the directory to index")
    index_parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_DIR), help="Path to local Qdrant database folder"
    )
    index_parser.add_argument(
        "--collection", type=str, default=DEFAULT_COLLECTION, help="Name of Qdrant collection"
    )

    # Command: search
    search_parser = subparsers.add_parser("search", help="Search the indexed codebase with a natural language query")
    search_parser.add_argument("query", type=str, help="Search query (e.g., 'calculate vram usage')")
    search_parser.add_argument(
        "--limit", "-n", type=int, default=5, help="Number of results to display (default: 5)"
    )
    search_parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_DIR), help="Path to local Qdrant database folder"
    )
    search_parser.add_argument(
        "--collection", type=str, default=DEFAULT_COLLECTION, help="Name of Qdrant collection"
    )
    search_parser.add_argument(
        "--generate", "-g", action="store_true", help="Generate an AI answer using the retrieved code context"
    )
    search_parser.add_argument(
        "--model", "-m", type=str, default="llama3", help="Ollama model to use for generation (default: llama3)"
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args.path, db_path=Path(args.db), collection_name=args.collection)
    elif args.command == "search":
        cmd_search(
            args.query,
            limit=args.limit,
            db_path=Path(args.db),
            collection_name=args.collection,
            generate=args.generate,
            model=args.model,
        )


if __name__ == "__main__":
    main()
