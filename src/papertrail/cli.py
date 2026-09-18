"""PaperTrail CLI – powered by Click + Rich."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

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
    "Index & Data": ["ingest", "arxiv", "discover", "list", "trends", "reset"],
    "Search & Analysis": ["search", "ask", "report", "digest", "intern"],
    "Scientist Tools": ["plan", "hypothesis", "experiment"],
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


def parse_paper_selection(selection_str: str, max_count: int) -> List[int]:
    """
    Parses user selection string like '1, 3-5' into 0-based integer indices.
    Validates range [1, max_count].
    Returns list of 0-based indices sorted and deduplicated.
    """
    s = selection_str.strip().lower()
    if not s or s in ("q", "quit", "exit", "cancel"):
        return []
    if s in ("all", "*"):
        return list(range(max_count))

    indices = set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for part in parts:
        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise ValueError(f"Invalid range format: '{part}'")
            try:
                start, end = int(bounds[0].strip()), int(bounds[1].strip())
            except ValueError:
                raise ValueError(f"Non-numeric range in '{part}'")
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                if 1 <= idx <= max_count:
                    indices.add(idx - 1)
                else:
                    raise ValueError(f"Number {idx} out of range (1-{max_count})")
        else:
            try:
                idx = int(part)
            except ValueError:
                raise ValueError(f"Invalid paper number: '{part}'")
            if 1 <= idx <= max_count:
                indices.add(idx - 1)
            else:
                raise ValueError(f"Number {idx} out of range (1-{max_count})")
    return sorted(indices)


def _search_and_select_arxiv(
    query: Optional[str] = None,
    limit: int = 10,
    categories: Optional[str] = None,
    sort: str = "relevance",
    download_all: bool = False,
    select: Optional[str] = None,
    no_clean: bool = False,
):
    """Core logic to search arXiv and interactively select papers to download."""
    from papertrail.ingestion.arxiv_client import ArxivClient
    from papertrail.ingestion.pipeline import index_single_paper
    from papertrail.retrieval.vectorstore import VectorStore

    cats = [c.strip() for c in categories.split(",") if c.strip()] if categories else []

    if not query or not query.strip():
        try:
            query = click.prompt("Enter search query for arXiv", type=str).strip()
        except (click.Abort, EOFError):
            console.print("\nCancelled.")
            return

    console.print(Panel(
        f"Query: [bold cyan]{query}[/bold cyan]\n"
        f"Limit: {limit} | Sort: {sort}" + (f" | Categories: {', '.join(cats)}" if cats else ""),
        title="arXiv Search",
        border_style="blue",
    ))

    with Progress(SpinnerColumn(), TextColumn("Searching arXiv..."), console=console, transient=True):
        client = ArxivClient(query=query, categories=cats, max_results=limit, sort_by=sort)
        try:
            papers = client.fetch_papers()
        except Exception as exc:
            console.print(f"[red]Error fetching from arXiv: {exc}[/red]")
            return

    if not papers:
        console.print(Panel(f"No papers found on arXiv for: {query}", title="arXiv Search", border_style="yellow"))
        return

    store = VectorStore()
    indexed_ids = set(store.indexed_paper_ids())

    table = Table(title=f"arXiv Results for '{query}' ({len(papers)} papers)", show_lines=True)
    table.add_column("#", style="bold cyan", width=3, justify="right")
    table.add_column("Status", width=10)
    table.add_column("arXiv ID", style="magenta", width=14)
    table.add_column("Title")
    table.add_column("Authors", width=25)
    table.add_column("Date", width=10)

    for i, p in enumerate(papers, 1):
        status = "[green]Indexed[/green]" if p.arxiv_id in indexed_ids else "[yellow]New[/yellow]"
        authors_str = ", ".join(p.authors[:2]) + ("..." if len(p.authors) > 2 else "")
        date_str = str(p.published.date()) if hasattr(p.published, "date") else (str(p.published)[:10] if p.published else "")
        table.add_row(str(i), status, p.arxiv_id, p.title[:75], authors_str, date_str)

    console.print(table)

    # Determine selection
    selected_indices: List[int] = []
    if download_all:
        selected_indices = list(range(len(papers)))
    elif select:
        try:
            selected_indices = parse_paper_selection(select, len(papers))
        except ValueError as exc:
            console.print(f"[red]Selection error: {exc}[/red]")
            return
    else:
        while True:
            console.print(
                "\n[bold]Select papers to download & index[/bold] "
                "(e.g. [cyan]1, 3-5[/cyan], [cyan]all[/cyan], [cyan]a 1[/cyan] for abstract, or [cyan]q[/cyan] to quit):"
            )
            try:
                choice = click.prompt("Selection", default="q", show_default=False).strip()
            except (click.Abort, EOFError):
                console.print("\nCancelled.")
                return

            if not choice or choice.lower() in ("q", "quit", "exit"):
                console.print("Cancelled. No papers downloaded.")
                return

            # Abstract inspection
            if choice.lower().startswith("a ") or choice.lower().startswith("abstract "):
                parts = choice.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    num = int(parts[1])
                    if 1 <= num <= len(papers):
                        p = papers[num - 1]
                        console.print(Panel(
                            f"[bold]{p.title}[/bold]\n"
                            f"Authors: {', '.join(p.authors)}\n"
                            f"Date: {p.published}\n\n"
                            f"{p.abstract}",
                            title=f"Abstract: [{num}] {p.arxiv_id}",
                            border_style="green",
                        ))
                    else:
                        console.print(f"[red]Invalid paper number {num}. Range is 1-{len(papers)}[/red]")
                continue

            try:
                selected_indices = parse_paper_selection(choice, len(papers))
                break
            except ValueError as exc:
                console.print(f"[red]{exc}[/red]")

    if not selected_indices:
        console.print("No papers selected.")
        return

    console.print(f"\n[bold green]Downloading and indexing {len(selected_indices)} paper(s)...[/bold green]")
    success_count = 0
    total_new_chunks = 0

    for idx in selected_indices:
        p = papers[idx]
        console.print(f"[{idx + 1}/{len(papers)}] [bold]{p.title[:70]}[/bold] ({p.arxiv_id})")
        with Progress(SpinnerColumn(), TextColumn("Processing..."), console=console, transient=True):
            success, msg, n_chunks = index_single_paper(p, store=store, clean=not no_clean)

        if success:
            console.print(f"  [green]✓ {msg}[/green]")
            success_count += 1
            total_new_chunks += n_chunks
        else:
            console.print(f"  [dim]• {msg}[/dim]")

    console.print(Panel(
        f"Completed! Indexed {success_count} new paper(s) with {total_new_chunks} chunks.\n"
        f"Total in store: {store.total_chunks} chunks across {len(store.indexed_paper_ids())} papers.",
        title="Ingest Complete",
        border_style="green",
    ))


@cli.command()
@click.option("--categories", "-c", default="cs.AI,cs.LG,cs.CL", show_default=True,
              help="Comma-separated arXiv category filters.")
@click.option("--query", "-q", default=None,
              help="Optional search query to filter papers on arXiv.")
@click.option("--max-results", "-n", default=10, show_default=True,
              help="Maximum number of papers to fetch.")
@click.option("--no-clean", is_flag=True, default=False,
              help="Skip text cleaning step.")
def ingest(categories, query, max_results, no_clean):
    """Fetch arXiv papers, download PDFs, and index them."""
    from papertrail.ingestion.arxiv_client import ArxivClient
    from papertrail.ingestion.pipeline import index_single_paper
    from papertrail.retrieval.vectorstore import VectorStore

    cats = [c.strip() for c in categories.split(",") if c.strip()]
    console.print(Panel(
        f"Fetching up to {max_results} papers" + (f" for query '{query}'" if query else "") + f" from {', '.join(cats)}",
        title="Ingest",
        border_style="blue",
    ))

    client = ArxivClient(query=query, categories=cats, max_results=max_results)
    console.print("Fetching paper list from arXiv...")
    papers = client.fetch_papers()
    console.print(f"Found {len(papers)} papers.")

    store = VectorStore()
    new_count = 0
    total_chunks = 0

    for i, paper in enumerate(papers, 1):
        console.print(f"[{i}/{len(papers)}] {paper.title[:80]}")
        success, msg, n_chunks = index_single_paper(paper, store=store, clean=not no_clean)
        console.print(f"  {msg}")
        if success:
            new_count += 1
            total_chunks += n_chunks

    console.print(Panel(
        f"Done! Indexed {new_count} new papers. "
        f"Total: {store.total_chunks} chunks, {len(store.indexed_paper_ids())} papers.",
        title="Ingest",
        border_style="blue",
    ))


@cli.command(name="arxiv")
@click.argument("query", required=False, default=None)
@click.option("--limit", "-n", default=10, show_default=True,
              help="Maximum number of papers to fetch from arXiv.")
@click.option("--categories", "-c", default=None,
              help="Comma-separated arXiv categories (e.g. cs.AI,cs.LG).")
@click.option("--sort", default="relevance",
              type=click.Choice(["relevance", "submittedDate", "lastUpdatedDate"]),
              show_default=True, help="Sort results by relevance or date.")
@click.option("--download-all", is_flag=True, default=False,
              help="Download and index all retrieved papers without prompting.")
@click.option("--select", "-s", default=None,
              help="Paper numbers to download (e.g. '1, 3-5', 'all').")
@click.option("--no-clean", is_flag=True, default=False,
              help="Skip text cleaning during extraction.")
def arxiv_cmd(query, limit, categories, sort, download_all, select, no_clean):
    """Search arXiv and interactively select papers to download and index."""
    _search_and_select_arxiv(
        query=query,
        limit=limit,
        categories=categories,
        sort=sort,
        download_all=download_all,
        select=select,
        no_clean=no_clean,
    )


@cli.command(name="discover")
@click.argument("query", required=False, default=None)
@click.option("--limit", "-n", default=10, show_default=True,
              help="Maximum number of papers to fetch from arXiv.")
@click.option("--categories", "-c", default=None,
              help="Filter by arXiv categories.")
@click.option("--sort", default="relevance",
              type=click.Choice(["relevance", "submittedDate", "lastUpdatedDate"]),
              show_default=True)
@click.option("--download-all", is_flag=True, default=False,
              help="Download all retrieved papers without prompting.")
@click.option("--select", "-s", default=None,
              help="Paper numbers to download (e.g. '1, 3-5', 'all').")
@click.option("--no-clean", is_flag=True, default=False,
              help="Skip text cleaning during extraction.")
def discover_cmd(query, limit, categories, sort, download_all, select, no_clean):
    """Alias for 'arxiv' – discover and download papers from arXiv."""
    _search_and_select_arxiv(
        query=query,
        limit=limit,
        categories=categories,
        sort=sort,
        download_all=download_all,
        select=select,
        no_clean=no_clean,
    )


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, show_default=True, help="Number of results.")
@click.option("--rerank/--no-rerank", default=True, show_default=True,
              help="Apply reranking after retrieval.")
@click.option("--arxiv", "-a", is_flag=True, default=False,
              help="Search arXiv directly and select papers to download and index.")
@click.option("--limit", "-n", default=10, show_default=True,
              help="Maximum arXiv results to fetch when --arxiv is used.")
def search(query, top_k, rerank, arxiv, limit):
    """Search indexed papers (or arXiv directly with --arxiv)."""
    if arxiv:
        _search_and_select_arxiv(query=query, limit=limit)
        return

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
@click.option("--top-n", default=10, show_default=True, help="Number of recent papers to include in the digest.")
def digest(top_n):
    """Generate a digest of recent papers."""
    from papertrail.agents.research_agent import ResearchAgent

    agent = ResearchAgent()
    need_index(agent)

    with Progress(SpinnerColumn(), TextColumn("Generating digest..."), console=console, transient=True):
        summary = agent.generate_digest(k=top_n)

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


@cli.command(name="intern")
@click.argument("prompt", required=False, default=None)
@click.option("--investigate", "-i", is_flag=True, default=False,
              help="Run an end-to-end literature investigation on the given topic.")
@click.option("--collect", is_flag=True, default=False,
              help="Collect, download, and index key papers from arXiv.")
@click.option("--reproduce", is_flag=True, default=False,
              help="Assess reproduction feasibility and requirements.")
def intern_cmd(prompt, investigate, collect, reproduce):
    """Interact with the PaperTrail AI Research Intern (powered by PydanticAI)."""
    from papertrail.agents.intern_agent import ResearchIntern

    if not prompt or not prompt.strip():
        try:
            prompt = click.prompt("What would you like your Research Intern to do?", type=str).strip()
        except (click.Abort, EOFError):
            console.print("\nCancelled.")
            return

    intern = ResearchIntern()
    console.print(Panel(prompt, title="Research Intern Task", border_style="cyan"))

    with Progress(SpinnerColumn(), TextColumn("Research Intern working..."), console=console, transient=True):
        if collect:
            response = intern.collect_data(prompt)
        elif reproduce:
            response = intern.reproduce_study(prompt)
        elif investigate:
            response = intern.investigate(prompt)
        else:
            response = intern.run_sync(prompt)

    console.print(Panel(Markdown(response), title="Research Intern Report", border_style="green"))


@cli.command(name="assistant")
@click.argument("prompt", required=False, default=None)
@click.option("--compare", "-c", is_flag=True, default=False,
              help="Conduct an in-depth comparative analysis across papers.")
@click.option("--gaps", "-g", is_flag=True, default=False,
              help="Identify literature gaps, untested assumptions, and open questions.")
@click.option("--benchmark", "-b", is_flag=True, default=False,
              help="Extract standardized benchmark results and comparative scores.")
@click.option("--implement", is_flag=True, default=False,
              help="Generate an engineering implementation blueprint.")
@click.option("--dossier", "-d", is_flag=True, default=False,
              help="Compile a structured research dossier for the Principal Scientist.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output structured Pydantic model results as JSON.")
def assistant_cmd(prompt, compare, gaps, benchmark, implement, dossier, as_json):
    """Interact with the Senior AI Research Assistant (powered by PydanticAI)."""
    from papertrail.agents.assistant_agent import ResearchAssistant

    if not prompt or not prompt.strip():
        try:
            prompt = click.prompt("What research question or topic should the Assistant analyze?", type=str).strip()
        except (click.Abort, EOFError):
            console.print("\nCancelled.")
            return

    assistant = ResearchAssistant()
    console.print(Panel(prompt, title="Research Assistant Task", border_style="cyan"))

    with Progress(SpinnerColumn(), TextColumn("Research Assistant analyzing..."), console=console, transient=True):
        if dossier:
            res_dossier = assistant.prepare_dossier(prompt)
            if as_json:
                console.print(res_dossier.model_dump_json(indent=2))
            else:
                md_content = f"# Research Dossier: {res_dossier.topic}\n\n"
                md_content += f"## Executive Summary\n{res_dossier.executive_summary}\n\n"
                if res_dossier.papers_analyzed:
                    md_content += "## Papers Analyzed\n"
                    for p in res_dossier.papers_analyzed:
                        md_content += f"### {p.title} (`{p.paper_id}`)\n"
                        md_content += f"- **Methodology:** {p.methodology}\n"
                        if p.key_findings:
                            md_content += f"- **Key Findings:** {', '.join(p.key_findings)}\n"
                        if p.strengths:
                            md_content += f"- **Strengths:** {', '.join(p.strengths)}\n"
                        if p.limitations:
                            md_content += f"- **Limitations:** {', '.join(p.limitations)}\n\n"
                if res_dossier.comparisons:
                    md_content += "## Comparative Dimensions\n"
                    for comp in res_dossier.comparisons:
                        md_content += f"### Dimension: {comp.dimension}\n"
                        md_content += f"{comp.analysis}\n\n"
                if res_dossier.gap_analysis:
                    md_content += "## Gap Analysis\n"
                    for g in res_dossier.gap_analysis.identified_gaps:
                        md_content += f"- **Gap:** {g}\n"
                    for q in res_dossier.gap_analysis.open_questions:
                        md_content += f"- **Open Question:** {q}\n"
                console.print(Panel(Markdown(md_content), title="Research Dossier", border_style="green"))
            return

        if gaps:
            gap_res = assistant.analyze_gaps(prompt)
            if as_json:
                console.print(gap_res.model_dump_json(indent=2))
            else:
                md_content = f"# Gap Analysis: {gap_res.topic}\n\n"
                if gap_res.identified_gaps:
                    md_content += "### Identified Gaps & Deficiencies\n"
                    for g in gap_res.identified_gaps:
                        md_content += f"- {g}\n"
                if gap_res.untested_assumptions:
                    md_content += "\n### Untested Assumptions\n"
                    for a in gap_res.untested_assumptions:
                        md_content += f"- {a}\n"
                if gap_res.open_questions:
                    md_content += "\n### Open Research Questions\n"
                    for q in gap_res.open_questions:
                        md_content += f"- {q}\n"
                if gap_res.promising_directions:
                    md_content += "\n### Promising Directions\n"
                    for d in gap_res.promising_directions:
                        md_content += f"- {d}\n"
                console.print(Panel(Markdown(md_content), title="Gap Analysis", border_style="green"))
            return

        if benchmark:
            bench_res = assistant.run_benchmarks(prompt)
            if as_json:
                console.print(bench_res.model_dump_json(indent=2))
            else:
                md_content = f"# Benchmark Results: {bench_res.benchmark_name}\n\n"
                md_content += f"- **Task Type:** {bench_res.task_type}\n"
                md_content += f"- **Primary Metric:** {bench_res.metric_name}\n"
                if bench_res.model_scores:
                    md_content += "- **Scores:**\n"
                    for m, s in bench_res.model_scores.items():
                        md_content += f"  - {m}: {s}\n"
                md_content += f"- **Baseline Comparison:** {bench_res.baseline_comparison}\n"
                if bench_res.compute_resources:
                    md_content += f"- **Compute Notes:** {bench_res.compute_resources}\n"
                console.print(Panel(Markdown(md_content), title="Benchmark Evaluation", border_style="green"))
            return

        if implement:
            response = assistant.plan_implementation(prompt)
        elif compare:
            response = assistant.compare_papers(prompt)
        else:
            response = assistant.run_sync(prompt)

    console.print(Panel(Markdown(response), title="Research Assistant Synthesis", border_style="green"))


@cli.command(name="researcher")
@click.argument("prompt", required=False, default=None)
@click.option("--hypothesis", "-h", is_flag=True, default=False,
              help="Formulate a testable scientific hypothesis with falsification bounds.")
@click.option("--experiment", "-e", is_flag=True, default=False,
              help="Design a controlled empirical experiment protocol.")
@click.option("--interpret", "-i", is_flag=True, default=False,
              help="Interpret empirical results against theory or hypotheses.")
@click.option("--plan", "-p", is_flag=True, default=False,
              help="Formulate a structured research roadmap.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output structured HypothesisSpec or ExperimentSpec as JSON.")
def researcher_cmd(prompt, hypothesis, experiment, interpret, plan, as_json):
    """Interact directly with the Tier 3 AI Researcher (powered by PydanticAI)."""
    from papertrail.agents.scientist_agent import ResearcherAgent

    if not prompt or not prompt.strip():
        try:
            prompt = click.prompt("What research question or task should the Researcher address?", type=str).strip()
        except (click.Abort, EOFError):
            console.print("\nCancelled.")
            return

    researcher = ResearcherAgent()
    console.print(Panel(prompt, title="Researcher Task", border_style="blue"))

    with Progress(SpinnerColumn(), TextColumn("Researcher working..."), console=console, transient=True):
        if hypothesis:
            if as_json:
                spec = researcher.formulate_hypothesis_spec(prompt)
                console.print(spec.model_dump_json(indent=2))
                return
            response = researcher.generate_hypothesis(prompt)
        elif experiment:
            if as_json:
                exp_spec = researcher.design_experiment_spec(prompt)
                console.print(exp_spec.model_dump_json(indent=2))
                return
            response = researcher.design_experiment(prompt)
        elif interpret:
            response = researcher.interpret_results(prompt)
        elif plan:
            response = researcher.generate_plan(prompt)
        else:
            response = researcher.agent.run_sync(prompt).output

    console.print(Panel(Markdown(str(response)), title="Researcher Output", border_style="blue"))


@cli.command(name="lead")
@click.argument("prompt", required=False, default=None)
@click.option("--orchestrate", "-o", is_flag=True, default=False,
              help="Orchestrate a complete research campaign across all 4 tiers.")
@click.option("--review-quality", "-q", is_flag=True, default=False,
              help="Conduct an expert peer-review evaluation on paper quality.")
@click.option("--roadmap", "-r", is_flag=True, default=False,
              help="Generate a strategic multi-phase research roadmap.")
@click.option("--json", "as_json", is_flag=True, default=False,
              help="Output scorecard or roadmap as JSON.")
def lead_cmd(prompt, orchestrate, review_quality, roadmap, as_json):
    """Interact with the Senior Research Lead & PI (Chief User Interface, powered by PydanticAI)."""
    from papertrail.agents.lead_agent import LeadResearcher

    if not prompt or not prompt.strip():
        try:
            prompt = click.prompt("What research initiative or question would you like to direct?", type=str).strip()
        except (click.Abort, EOFError):
            console.print("\nCancelled.")
            return

    lead = LeadResearcher()
    console.print(Panel(prompt, title="Research Lead Directive", border_style="magenta"))

    with Progress(SpinnerColumn(), TextColumn("Research Lead orchestrating..."), console=console, transient=True):
        if review_quality:
            scorecard = lead.review_paper_quality(prompt)
            if as_json:
                console.print(scorecard.model_dump_json(indent=2))
            else:
                md = f"# Paper Review Scorecard: {scorecard.paper_title}\n\n"
                md += f"**Overall Verdict:** {scorecard.overall_verdict}\n\n"
                md += f"- **Soundness:** {scorecard.soundness_score}/5\n"
                md += f"- **Novelty:** {scorecard.novelty_score}/5\n"
                md += f"- **Empirical Rigor:** {scorecard.empirical_rigor_score}/5\n"
                md += f"- **Clarity:** {scorecard.clarity_score}/5\n\n"
                if scorecard.strengths:
                    md += "### Strengths\n"
                    for s in scorecard.strengths:
                        md += f"- {s}\n"
                if scorecard.weaknesses:
                    md += "\n### Weaknesses\n"
                    for w in scorecard.weaknesses:
                        md += f"- {w}\n"
                if scorecard.recommendations:
                    md += "\n### Actionable Recommendations\n"
                    for rec in scorecard.recommendations:
                        md += f"- {rec}\n"
                console.print(Panel(Markdown(md), title="Paper Quality Assessment", border_style="magenta"))
            return

        if roadmap:
            res_roadmap = lead.plan_roadmap(prompt)
            if as_json:
                console.print(res_roadmap.model_dump_json(indent=2))
            else:
                md = f"# Research Roadmap: {res_roadmap.initiative}\n\n"
                md += f"**Strategic Objective:** {res_roadmap.strategic_objective}\n\n"
                if res_roadmap.phases:
                    md += "## Phased Execution\n"
                    for phase in res_roadmap.phases:
                        for k, v in phase.items():
                            md += f"- **{k}:** {v}\n"
                if res_roadmap.delegation_plan:
                    md += "\n## Team Delegation Plan\n"
                    for role, assignment in res_roadmap.delegation_plan.items():
                        md += f"- **{role}:** {assignment}\n"
                if res_roadmap.critical_risks:
                    md += "\n## Critical Risks\n"
                    for r in res_roadmap.critical_risks:
                        md += f"- {r}\n"
                console.print(Panel(Markdown(md), title="Strategic Research Roadmap", border_style="magenta"))
            return

        if orchestrate:
            response = lead.orchestrate_campaign(prompt)
        else:
            response = lead.run_sync(prompt)

    console.print(Panel(Markdown(response), title="Research Lead Synthesis", border_style="magenta"))


@cli.command(name="labs")
def labs_cmd():
    """List all 10 PaperTrail research laboratories by category."""
    from papertrail.labs.registry import get_all_labs

    labs = get_all_labs()
    categories = {
        "domain": "Core Research Domains",
        "methodology": "Cross-Cutting Methodologies",
        "application": "Applications & Translation",
    }

    md = "# PaperTrail Research Laboratories\n\n"
    for cat_key, cat_title in categories.items():
        cat_labs = [l for l in labs if l.category == cat_key]
        if not cat_labs:
            continue
        md += f"## {cat_title}\n\n"
        for lab in cat_labs:
            md += f"### {lab.full_name} (`{lab.code}`)\n"
            md += f"**Focus:** {lab.focus}\n\n"
            md += f"**Key Topics:** {', '.join(lab.key_topics)}\n\n"

    console.print(Panel(Markdown(md), title="Research Laboratories Directory", border_style="cyan"))


@cli.command(name="lab")
@click.argument("lab_code")
@click.argument("prompt", required=False, default=None)
def lab_cmd(lab_code, prompt):
    """Interact directly with a specialized research laboratory (e.g. LMI, AI, FMPT, ISAI)."""
    from papertrail.labs.lab_agent import ResearchLabAgent
    from papertrail.labs.registry import get_lab

    lab_info = get_lab(lab_code)
    if not lab_info:
        console.print(f"[red]Error:[/red] Unknown research lab code '{lab_code}'. Run [bold]papertrail labs[/bold] to see valid codes.")
        return

    if not prompt or not prompt.strip():
        try:
            prompt = click.prompt(f"Inquiry for the {lab_info.full_name}", type=str).strip()
        except (click.Abort, EOFError):
            console.print("\nCancelled.")
            return

    agent = ResearchLabAgent(lab_code=lab_info.code)
    console.print(Panel(prompt, title=f"{lab_info.short_name} Inquiry", border_style="cyan"))

    with Progress(SpinnerColumn(), TextColumn(f"{lab_info.code} Lab analyzing..."), console=console, transient=True):
        response = agent.run_sync(prompt)

    console.print(Panel(Markdown(response), title=f"{lab_info.code} Laboratory Findings", border_style="green"))


@cli.command(name="matrix")
@click.argument("project_name")
@click.option("--domain", "-d", required=True, help="Primary domain lab code (e.g. AI, LMI, VI, RCI).")
@click.option("--methodology", "-m", default="FMPT", help="Comma-separated cross-cutting methodology labs (e.g. FMPT,ISAI).")
@click.option("--application", "-a", default="", help="Comma-separated application labs (e.g. AISL).")
@click.option("--mission", required=False, default=None, help="High-level mission statement.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output synthesis as JSON.")
def matrix_cmd(project_name, domain, methodology, application, mission, as_json):
    """Launch a cross-lab matrix project (e.g. Domain x Methodology x Application)."""
    from papertrail.schemas.schema import MatrixProject
    from papertrail.labs.collaboration import MatrixProjectCoordinator

    if not mission or not mission.strip():
        mission = f"Investigate cross-cutting breakthroughs at the intersection of {domain} and {methodology}."

    methodologies = [m.strip().upper() for m in methodology.split(",") if m.strip()]
    applications = [a.strip().upper() for a in application.split(",") if a.strip()]

    proj = MatrixProject(
        project_name=project_name,
        primary_domain=domain.strip().upper(),
        collaborating_methodologies=methodologies,
        collaborating_applications=applications,
        mission_statement=mission,
    )

    console.print(Panel(
        f"**Project:** {proj.project_name}\n"
        f"**Matrix:** `{proj.primary_domain}` × `{', '.join(proj.collaborating_methodologies)}`"
        + (f" × `{', '.join(proj.collaborating_applications)}`" if proj.collaborating_applications else "") + "\n"
        f"**Mission:** {proj.mission_statement}",
        title="Cross-Lab Matrix Initiative",
        border_style="magenta",
    ))

    coord = MatrixProjectCoordinator()
    with Progress(SpinnerColumn(), TextColumn("Coordinating multi-lab matrix project..."), console=console, transient=True):
        synthesis = coord.execute_matrix_project(proj)

    if as_json:
        console.print(synthesis.model_dump_json(indent=2))
    else:
        md = f"# Cross-Lab Synthesis: {synthesis.project_name}\n\n"
        md += f"**Participating Labs:** {', '.join(synthesis.participating_labs)}\n\n"
        md += f"## Executive Summary\n{synthesis.executive_summary}\n\n"
        md += f"## Domain Breakthroughs ({proj.primary_domain})\n{synthesis.domain_insights}\n\n"
        md += f"## Methodology & Systems Specifications\n{synthesis.methodology_specifications}\n\n"
        if synthesis.application_impact:
            md += f"## Downstream Application Impact\n{synthesis.application_impact}\n\n"
        if synthesis.cross_cutting_synergies:
            md += "## Cross-Cutting Synergies\n"
            for syn in synthesis.cross_cutting_synergies:
                md += f"- {syn}\n"

        console.print(Panel(Markdown(md), title="Matrix Project Report", border_style="magenta"))


@cli.command()
@click.argument("topic")
def plan(topic):
    """Generate a research plan using the AI Scientist."""
    from papertrail.agents.scientist_agent import ScientistAgent

    agent = ScientistAgent()
    console.print(Panel(topic, title="Research Topic", border_style="blue"))
    with Progress(SpinnerColumn(), TextColumn("Scientist planning..."), console=console, transient=True):
        output = agent.generate_plan(topic)
    console.print(Panel(Markdown(output), title="Research Plan", border_style="blue"))


@cli.command()
@click.argument("topic")
def hypothesis(topic):
    """Formulate novel research hypotheses using the AI Scientist."""
    from papertrail.agents.scientist_agent import ScientistAgent

    agent = ScientistAgent()
    console.print(Panel(topic, title="Topic", border_style="blue"))
    with Progress(SpinnerColumn(), TextColumn("Scientist hypothesizing..."), console=console, transient=True):
        output = agent.generate_hypothesis(topic)
    console.print(Panel(Markdown(output), title="Hypotheses", border_style="blue"))


@cli.command()
@click.argument("hypothesis_text")
def experiment(hypothesis_text):
    """Design empirical experiments for a hypothesis using the AI Scientist."""
    from papertrail.agents.scientist_agent import ScientistAgent

    agent = ScientistAgent()
    console.print(Panel(hypothesis_text, title="Hypothesis", border_style="blue"))
    with Progress(SpinnerColumn(), TextColumn("Designing experiment..."), console=console, transient=True):
        output = agent.design_experiment(hypothesis_text)
    console.print(Panel(Markdown(output), title="Experiment Design", border_style="blue"))



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
