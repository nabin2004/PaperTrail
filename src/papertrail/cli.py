"""
PaperTrail CLI – powered by Click + Rich.

Commands
--------
  ingest   Fetch papers from arXiv, download PDFs, embed + index
  search   Semantic search over indexed papers
  ask      RAG-based Q&A against indexed papers
  report   Full research report (plan → retrieve → synthesise → critique)
  list     List all indexed papers
  trends   Show trending keywords and categories
  reset    Wipe the FAISS index and all stored data
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


# ──────────────────────────────────────────────────────────────
# CLI group
# ──────────────────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0", prog_name="papertrail")
def cli() -> None:
    """PaperTrail – AI-powered research paper discovery and synthesis."""
    _load_dotenv()


# ──────────────────────────────────────────────────────────────
# ingest
# ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--categories", "-c", default="cs.AI,cs.LG,cs.CL", show_default=True,
              help="Comma-separated arXiv category filters.")
@click.option("--max-results", "-n", default=10, show_default=True,
              help="Maximum number of papers to fetch.")
@click.option("--no-clean", is_flag=True, default=False,
              help="Skip text cleaning step.")
def ingest(categories: str, max_results: int, no_clean: bool) -> None:
    """Fetch arXiv papers, download PDFs, process and index them."""
    from papertrail.ingestion.arxiv_client import ArxivClient
    from papertrail.ingestion.metadata import metadata_exists, save_metadata
    from papertrail.ingestion.pdf_loader import download_pdf, extract_text, save_processed_text
    from papertrail.processing.cleaners import clean_text
    from papertrail.processing.embeddings import embed_texts
    from papertrail.processing.splitters import split_and_save
    from papertrail.retrieval.vectorstore import VectorStore
    import json

    cats = [c.strip() for c in categories.split(",") if c.strip()]
    console.print(Panel(
        f"Fetching up to [bold]{max_results}[/bold] papers from {', '.join(cats)}",
        title="[green]PaperTrail – Ingest[/green]"
    ))

    client = ArxivClient(categories=cats, max_results=max_results)

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        t = prog.add_task("Fetching paper list from arXiv…", total=None)
        papers = client.fetch_papers()
        prog.update(t, description=f"Found [bold]{len(papers)}[/bold] papers.")

    store = VectorStore()
    new_count = 0

    for i, paper in enumerate(papers, 1):
        console.print(f"[{i}/{len(papers)}] [cyan]{paper.title[:80]}[/cyan]")

        # Skip already-indexed papers
        if paper.arxiv_id in store.indexed_paper_ids():
            console.print("  [dim]already indexed – skipping[/dim]")
            continue

        try:
            with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as p2:
                t2 = p2.add_task("  Downloading PDF…", total=None)
                pdf_path = download_pdf(paper)
                if pdf_path is None:
                    console.print("  [red]No PDF URL – skipping[/red]")
                    continue

                p2.update(t2, description="  Extracting text…")
                text_path = save_processed_text(pdf_path, clean=not no_clean)

                p2.update(t2, description="  Splitting into chunks…")
                chunks_path = split_and_save(text_path)

                p2.update(t2, description="  Creating embeddings…")
                chunk_dicts = _load_chunks(chunks_path)
                if not chunk_dicts:
                    continue
                texts = [c["text"] for c in chunk_dicts]
                embeddings = embed_texts(texts)

                p2.update(t2, description="  Saving to index…")
                store.add_chunks(embeddings, chunk_dicts)

            # Persist metadata
            paper.indexed = True
            save_metadata(paper)
            new_count += 1
            console.print(f"  [green]✓ indexed {len(chunk_dicts)} chunks[/green]")

        except Exception as exc:
            console.print(f"  [red]Error: {exc}[/red]")

    console.print(f"\n[bold green]Done.[/bold green] Indexed {new_count} new papers. "
                  f"Total in store: {store.total_chunks} chunks across {len(store.indexed_paper_ids())} papers.")


# ──────────────────────────────────────────────────────────────
# search
# ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, show_default=True, help="Number of results.")
@click.option("--rerank/--no-rerank", default=True, show_default=True,
              help="Apply reranking after retrieval.")
def search(query: str, top_k: int, rerank: bool) -> None:
    """Semantic search over indexed papers."""
    from papertrail.retrieval.retrievers import PaperRetriever
    from papertrail.retrieval.reranker import rerank as do_rerank

    retriever = PaperRetriever()
    if retriever.is_empty():
        console.print("[red]No papers indexed yet. Run [bold]papertrail ingest[/bold] first.[/red]")
        sys.exit(1)

    with Progress(SpinnerColumn(), TextColumn("Searching…"), console=console, transient=True):
        results = retriever.retrieve(query, k=top_k * 3)
        if rerank:
            results = do_rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f"Top {len(results)} results for: [italic]{query}[/italic]",
                  show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=6)
    table.add_column("Paper", style="cyan")
    table.add_column("Excerpt")

    for i, r in enumerate(results, 1):
        title = (r.title or r.paper_id)[:50]
        authors = (", ".join(r.authors[:2]) + "…") if r.authors else ""
        caption = f"{title}\n[dim]{authors} {r.published or ''}[/dim]"
        table.add_row(str(i), f"{r.score:.3f}", caption, r.text[:200] + "…")

    console.print(table)


# ──────────────────────────────────────────────────────────────
# ask
# ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
@click.option("--top-k", "-k", default=6, show_default=True,
              help="Number of chunks used for answering.")
def ask(question: str, top_k: int) -> None:
    """Ask a question and get a synthesized answer from indexed papers."""
    from papertrail.agents.research_agent import ResearchAgent

    agent = ResearchAgent(rerank_k=top_k)
    if not agent.is_ready():
        console.print("[red]No papers indexed yet. Run [bold]papertrail ingest[/bold] first.[/red]")
        sys.exit(1)

    console.print(Panel(f"[bold]{question}[/bold]", title="Question"))

    with Progress(SpinnerColumn(), TextColumn("Thinking…"), console=console, transient=True):
        answer = agent.ask(question)

    console.print(Panel(Markdown(answer), title="[green]Answer[/green]", expand=False))


# ──────────────────────────────────────────────────────────────
# report
# ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
@click.option("--top-k", "-k", default=6, show_default=True,
              help="Number of chunks used for synthesis.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Optional path to save the report as Markdown.")
def report(question: str, top_k: int, output: str | None) -> None:
    """Generate a full research report: plan → retrieve → synthesise → critique → evaluate."""
    from papertrail.agents.research_agent import ResearchAgent

    agent = ResearchAgent(rerank_k=top_k)
    if not agent.is_ready():
        console.print("[red]No papers indexed yet. Run [bold]papertrail ingest[/bold] first.[/red]")
        sys.exit(1)

    console.print(Panel(f"[bold]{question}[/bold]", title="[green]Research Question[/green]"))

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as prog:
        t = prog.add_task("Running research pipeline…", total=None)
        rep = agent.research(question)
        prog.update(t, description="Done.")

    # ── Display ───────────────────────────────────────────────
    console.rule("[bold green]Research Plan[/bold green]")
    if rep.plan:
        console.print(f"[bold]Queries:[/bold] {', '.join(rep.plan.queries or [question])}")
        console.print(f"[bold]Concepts:[/bold] {', '.join(rep.plan.concepts or [])}")

    console.rule("[bold green]Synthesis[/bold green]")
    console.print(Markdown(rep.synthesis))

    console.rule("[bold yellow]Critique[/bold yellow]")
    console.print(Markdown(rep.critique))

    console.rule("[bold blue]Sources[/bold blue]")
    for i, src in enumerate(rep.sources, 1):
        console.print(f"[{i}] [cyan]{src.title or src.paper_id}[/cyan] – score {src.score:.3f}")

    console.rule("[bold]Evaluation[/bold]")
    console.print(f"Faithfulness : {rep.faithfulness_score:.2%}")
    console.print(f"Coverage     : {rep.coverage_score:.2%}")

    # ── Save ──────────────────────────────────────────────────
    if output:
        md = _report_to_markdown(rep)
        Path(output).write_text(md, encoding="utf-8")
        console.print(f"\n[green]Report saved → {output}[/green]")


# ──────────────────────────────────────────────────────────────
# list
# ──────────────────────────────────────────────────────────────

@cli.command(name="list")
def list_papers() -> None:
    """List all indexed papers."""
    from papertrail.ingestion.metadata import load_all_metadata

    papers = load_all_metadata()
    if not papers:
        console.print("[yellow]No papers indexed yet.[/yellow]")
        return

    table = Table(title=f"{len(papers)} Indexed Papers", show_lines=True)
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("Title")
    table.add_column("Authors", width=30)
    table.add_column("Date", width=12)
    table.add_column("Category", width=10)

    for p in papers:
        authors = ", ".join(p.authors[:2]) + ("…" if len(p.authors) > 2 else "")
        table.add_row(
            p.arxiv_id,
            p.title[:60],
            authors,
            str(p.published.date()),
            p.primary_category,
        )

    console.print(table)


# ──────────────────────────────────────────────────────────────
# trends
# ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--top-n", default=20, show_default=True, help="Number of keywords to show.")
def trends(top_n: int) -> None:
    """Show trending keywords and categories from all indexed papers."""
    from papertrail.ingestion.metadata import load_all_metadata
    from papertrail.memory.trend_memory import TrendMemory

    papers = load_all_metadata()
    if not papers:
        console.print("[yellow]No papers indexed yet.[/yellow]")
        return

    mem = TrendMemory()
    mem.update(papers)

    kw_table = Table(title=f"Top {top_n} Keywords", show_lines=False, box=None)
    kw_table.add_column("Keyword", style="cyan")
    kw_table.add_column("Count", justify="right")
    for kw, cnt in mem.top_keywords(top_n):
        kw_table.add_row(kw, str(cnt))

    cat_table = Table(title="Top Categories", show_lines=False, box=None)
    cat_table.add_column("Category", style="green")
    cat_table.add_column("Count", justify="right")
    for cat, cnt in mem.top_categories(10):
        cat_table.add_row(cat, str(cnt))

    console.print(kw_table)
    console.print(cat_table)


# ──────────────────────────────────────────────────────────────
# reset
# ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def reset(yes: bool) -> None:
    """Wipe the FAISS index and all stored metadata/chunks."""
    if not yes:
        click.confirm("This will delete ALL indexed data. Continue?", abort=True)

    from papertrail.retrieval.vectorstore import VectorStore
    import shutil, os

    store = VectorStore()
    store.reset()
    console.print("[green]FAISS index cleared.[/green]")

    _data = Path(os.getenv("DATA_DIR", "data"))
    for subdir in ("metadata", "chunks", "processed"):
        d = _data / subdir
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]Cleared {d}[/green]")

    console.print("[bold green]Reset complete.[/bold green]")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _load_chunks(chunks_path: Path) -> list:
    import json
    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def _report_to_markdown(rep) -> str:
    lines = [
        f"# Research Report\n",
        f"**Question:** {rep.question}\n",
        f"**Generated:** {rep.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n",
        f"\n---\n",
        f"## Synthesis\n\n{rep.synthesis}\n",
        f"\n## Critique\n\n{rep.critique}\n",
        f"\n## Sources\n",
    ]
    for i, src in enumerate(rep.sources, 1):
        lines.append(f"{i}. **{src.title or src.paper_id}** (score: {src.score:.3f})\n")
    lines += [
        f"\n## Evaluation\n",
        f"- **Faithfulness:** {rep.faithfulness_score:.2%}\n",
        f"- **Coverage:** {rep.coverage_score:.2%}\n",
    ]
    return "".join(lines)


if __name__ == "__main__":
    cli()
