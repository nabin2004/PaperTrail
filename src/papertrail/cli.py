"""PaperTrail CLI – powered by Click + Rich."""
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

## Makes cli command faster in windows by avoiding encoding issues with stdout/stderr
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

console = Console()

# add newer commands here to groupp them in the help output
COMMAND_GROUPS = {
    "Index & Data": ["ingest", "list", "trends", "reset"],
    "Search & Analysis": ["search", "ask", "report", "digest"],
}


class PaperTrailGroup(click.Group):
    def format_help(self, ctx, formatter):
        console.print(Panel(self.help, title="PaperTrail", border_style="blue"))
        for group, names in COMMAND_GROUPS.items():
            rows = []
            for name in names:
                cmd = self.get_command(ctx, name)
                if cmd:
                    rows.append(f"  {name:<12}  {cmd.get_short_help_str()}")
            if rows:
                console.print(Panel("\n".join(rows), title=group, border_style="blue"))


@click.group(cls=PaperTrailGroup)
@click.version_option("0.1.0", prog_name="papertrail")
def cli():
    """AI-powered research paper discovery and synthesis."""
    _load_dotenv()


def need_index(obj):
    empty = obj.is_empty() if hasattr(obj, "is_empty") else not obj.is_ready()
    if empty:
        console.print("[red]No papers yet - run: papertrail ingest[/red]")
        sys.exit(1)


@cli.command()
@click.option("--categories", "-c", default="cs.AI,cs.LG,cs.CL", show_default=True,
              help="Comma-separated arXiv category filters.")
@click.option("--max-results", "-n", default=10, show_default=True,
              help="Maximum number of papers to fetch.")
@click.option("--no-clean", is_flag=True, default=False,
              help="Skip text cleaning step.")
def ingest(categories, max_results, no_clean):
    """Fetch arXiv papers, download PDFs, and index them."""
    from papertrail.ingestion.arxiv_client import ArxivClient
    from papertrail.ingestion.metadata import save_metadata
    from papertrail.ingestion.pdf_loader import download_pdf, save_processed_text
    from papertrail.processing.embeddings import embed_texts
    from papertrail.processing.splitters import split_and_save
    from papertrail.retrieval.vectorstore import VectorStore

    cats = [c.strip() for c in categories.split(",") if c.strip()]
    console.print(Panel(
        f"Fetching up to {max_results} papers from {', '.join(cats)}",
        title="Ingest",
        border_style="blue",
    ))

    client = ArxivClient(categories=cats, max_results=max_results)
    console.print("Fetching paper list from arXiv...")
    papers = client.fetch_papers()
    console.print(f"Found {len(papers)} papers.")

    store = VectorStore()
    new_count = 0

    for i, paper in enumerate(papers, 1):
        console.print(f"[{i}/{len(papers)}] {paper.title[:80]}")

        if paper.arxiv_id in store.indexed_paper_ids():
            console.print("  already indexed, skipping")
            continue

        try:
            console.print("  downloading...")
            pdf_path = download_pdf(paper)
            if pdf_path is None:
                console.print("  no PDF, skipping")
                continue

            console.print("  extracting text...")
            text_path = save_processed_text(pdf_path, clean=not no_clean)

            console.print("  splitting into chunks...")
            chunks_path = split_and_save(text_path)

            console.print("  creating embeddings...")
            chunk_dicts = _load_chunks(chunks_path)
            if not chunk_dicts:
                continue
            texts = [c["text"] for c in chunk_dicts]
            embeddings = embed_texts(texts)

            console.print("  saving to index...")
            store.add_chunks(embeddings, chunk_dicts)

            paper.indexed = True
            save_metadata(paper)
            new_count += 1
            console.print(f"  done - {len(chunk_dicts)} chunks")

        except Exception as exc:
            console.print(f"  error: {exc}")

    console.print(Panel(
        f"Done! Indexed {new_count} new papers. "
        f"Total: {store.total_chunks} chunks, {len(store.indexed_paper_ids())} papers.",
        title="Ingest",
        border_style="blue",
    ))


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, show_default=True, help="Number of results.")
@click.option("--rerank/--no-rerank", default=True, show_default=True,
              help="Apply reranking after retrieval.")
def search(query, top_k, rerank):
    """Search indexed papers."""
    from papertrail.retrieval.retrievers import PaperRetriever
    from papertrail.retrieval.reranker import rerank as do_rerank

    retriever = PaperRetriever()
    need_index(retriever)

    with Progress(SpinnerColumn(), TextColumn("Searching..."), console=console, transient=True):
        results = retriever.retrieve(query, k=top_k * 3)
        if rerank:
            results = do_rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]

    if not results:
        console.print(Panel("No results found.", title="Search", border_style="blue"))
        return

    table = Table(title=f"Results for: {query}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=6)
    table.add_column("Paper", style="cyan")
    table.add_column("Excerpt")

    for i, r in enumerate(results, 1):
        title = (r.title or r.paper_id)[:50]
        authors = (", ".join(r.authors[:2]) + "...") if r.authors else ""
        caption = f"{title}\n{authors} {r.published or ''}"
        table.add_row(str(i), f"{r.score:.3f}", caption, r.text[:200] + "...")

    console.print(Panel(table, title="Search", border_style="blue"))


@cli.command()
@click.argument("question")
@click.option("--top-k", "-k", default=6, show_default=True,
              help="Number of chunks used for answering.")
def ask(question, top_k):
    """Ask a question about indexed papers."""
    from papertrail.agents.research_agent import ResearchAgent

    agent = ResearchAgent(rerank_k=top_k)
    need_index(agent)

    console.print(Panel(question, title="Question", border_style="blue"))

    with Progress(SpinnerColumn(), TextColumn("Thinking..."), console=console, transient=True):
        answer = agent.ask(question)

    console.print(Panel(Markdown(answer), title="Answer", border_style="blue", expand=False))


@cli.command()
@click.argument("question")
@click.option("--top-k", "-k", default=6, show_default=True,
              help="Number of chunks used for synthesis.")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Optional path to save the report as Markdown.")
def report(question, top_k, output):
    """Generate a research report."""
    from papertrail.agents.research_agent import ResearchAgent

    agent = ResearchAgent(rerank_k=top_k)
    need_index(agent)

    console.print(Panel(question, title="Research Question", border_style="blue"))

    with Progress(SpinnerColumn(), TextColumn("Running..."), console=console, transient=True):
        rep = agent.research(question)

    if rep.plan:
        console.print(Panel(
            Markdown(
                f"**Queries:** {', '.join(rep.plan.queries or [question])}\n\n"
                f"**Concepts:** {', '.join(rep.plan.concepts or [])}"
            ),
            title="Research Plan",
            border_style="blue",
        ))

    console.print(Panel(Markdown(rep.synthesis), title="Synthesis", border_style="blue"))
    console.print(Panel(Markdown(rep.critique), title="Critique", border_style="blue"))
    console.print(Panel(
        "\n".join(f"{i}. {s.title or s.paper_id} - score {s.score:.3f}" for i, s in enumerate(rep.sources, 1)),
        title="Sources",
        border_style="blue",
    ))
    console.print(Panel(
        f"Faithfulness: {rep.faithfulness_score:.2%}\nCoverage: {rep.coverage_score:.2%}",
        title="Evaluation",
        border_style="blue",
    ))

    if output:
        sources_md = "\n".join(
            f"{i}. **{s.title or s.paper_id}** (score: {s.score:.3f})"
            for i, s in enumerate(rep.sources, 1)
        )
        md = f"""# Research Report

**Question:** {rep.question}
**Generated:** {rep.created_at.strftime('%Y-%m-%d %H:%M UTC')}

---

## Synthesis

{rep.synthesis}

## Critique

{rep.critique}

## Sources

{sources_md}

## Evaluation

- **Faithfulness:** {rep.faithfulness_score:.2%}
- **Coverage:** {rep.coverage_score:.2%}
"""
        Path(output).write_text(md, encoding="utf-8")
        console.print(Panel(f"Saved to {output}", title="Report", border_style="blue"))


@cli.command(name="list")
def list_papers():
    """List indexed papers."""
    from papertrail.ingestion.metadata import load_all_metadata

    papers = load_all_metadata()
    if not papers:
        console.print(Panel("No papers indexed yet.", title="Papers", border_style="blue"))
        return

    table = Table(title=f"{len(papers)} papers", show_lines=True)
    table.add_column("arXiv ID", style="cyan", width=14)
    table.add_column("Title")
    table.add_column("Authors", width=30)
    table.add_column("Date", width=12)
    table.add_column("Category", width=10)

    for p in papers:
        authors = ", ".join(p.authors[:2]) + ("..." if len(p.authors) > 2 else "")
        table.add_row(p.arxiv_id, p.title[:60], authors, str(p.published.date()), p.primary_category)

    console.print(Panel(table, title="Papers", border_style="blue"))


@cli.command()
@click.option("--top-n", default=20, show_default=True, help="Number of keywords to show.")
def trends(top_n):
    """Show trending keywords and categories."""
    from papertrail.ingestion.metadata import load_all_metadata
    from papertrail.memory.trend_memory import TrendMemory

    papers = load_all_metadata()
    if not papers:
        console.print(Panel("No papers indexed yet.", title="Trends", border_style="blue"))
        return

    mem = TrendMemory()
    mem.update(papers)

    kw_table = Table(title=f"Top {top_n} keywords", show_lines=False, box=None)
    kw_table.add_column("Keyword", style="cyan")
    kw_table.add_column("Count", justify="right")
    for kw, cnt in mem.top_keywords(top_n):
        kw_table.add_row(kw, str(cnt))

    cat_table = Table(title="Top categories", show_lines=False, box=None)
    cat_table.add_column("Category", style="green")
    cat_table.add_column("Count", justify="right")
    for cat, cnt in mem.top_categories(10):
        cat_table.add_row(cat, str(cnt))

    console.print(Panel(kw_table, title="Keywords", border_style="blue"))
    console.print(Panel(cat_table, title="Categories", border_style="blue"))


@cli.command()
def digest():
    """Generate a digest of recent papers."""
    from papertrail.agents.research_agent import ResearchAgent

    agent = ResearchAgent()
    need_index(agent)

    with Progress(SpinnerColumn(), TextColumn("Generating digest..."), console=console, transient=True):
        summary = agent.generate_digest()

    console.print(Panel(Markdown(summary), title="Digest", border_style="blue", expand=False))


@cli.command()
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def reset(yes):
    """Wipe the index and stored data."""
    if not yes:
        click.confirm("This will delete ALL indexed data. Continue?", abort=True)

    import os
    import shutil
    from papertrail.retrieval.vectorstore import VectorStore

    store = VectorStore()
    store.reset()

    data_dir = Path(os.getenv("DATA_DIR", "data"))
    msg = "FAISS index cleared."
    for subdir in ("metadata", "chunks", "processed"):
        d = data_dir / subdir
        if d.exists():
            shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            msg += f"\nCleared {d}"

    console.print(Panel(msg, title="Reset", border_style="blue"))


def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


def _load_chunks(chunks_path):
    import json
    chunks = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


if __name__ == "__main__":
    cli()
