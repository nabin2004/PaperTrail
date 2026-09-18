"""
Unit tests for arXiv search, selection parsing, and ingestion pipeline.
"""
import pytest
from unittest.mock import MagicMock, patch
from click.testing import CliRunner

from papertrail.cli import cli, parse_paper_selection
from papertrail.ingestion.arxiv_client import ArxivClient
from papertrail.schemas.schema import Paper


SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v7</id>
    <title>
      Attention Is
      All You Need
    </title>
    <summary>
      The dominant sequence transduction models are based on complex recurrent or
      convolutional neural networks.
    </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <published>2017-06-12T17:57:34Z</published>
    <updated>2023-08-02T01:07:07Z</updated>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
    <link href="http://arxiv.org/pdf/1706.03762v7" title="pdf" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


# ── Selection Parsing ────────────────────────────────────────────────────────

def test_parse_paper_selection_valid():
    assert parse_paper_selection("1", 5) == [0]
    assert parse_paper_selection("1, 3", 5) == [0, 2]
    assert parse_paper_selection("1-3", 5) == [0, 1, 2]
    assert parse_paper_selection("1, 3-5", 5) == [0, 2, 3, 4]
    assert parse_paper_selection("5-3", 5) == [2, 3, 4]
    assert parse_paper_selection("all", 4) == [0, 1, 2, 3]
    assert parse_paper_selection("*", 3) == [0, 1, 2]
    assert parse_paper_selection("q", 5) == []
    assert parse_paper_selection("quit", 5) == []
    assert parse_paper_selection("", 5) == []


def test_parse_paper_selection_invalid():
    with pytest.raises(ValueError, match="out of range"):
        parse_paper_selection("6", 5)

    with pytest.raises(ValueError, match="out of range"):
        parse_paper_selection("0", 5)

    with pytest.raises(ValueError, match="Invalid"):
        parse_paper_selection("abc", 5)

    with pytest.raises(ValueError, match="Invalid"):
        parse_paper_selection("1-2-3", 5)


# ── ArxivClient ──────────────────────────────────────────────────────────────

def test_arxiv_client_url_query():
    client = ArxivClient(query="transformer attention mechanism", max_results=5)
    url = client.build_query_url()
    assert "export.arxiv.org/api/query" in url
    assert "all%3Atransformer+attention+mechanism" in url or "all:transformer+attention+mechanism" in url
    assert "max_results=5" in url
    assert "sortBy=relevance" in url


def test_arxiv_client_url_categories():
    client = ArxivClient(categories=["cs.AI", "cs.LG"], max_results=10)
    url = client.build_query_url()
    assert "cat:cs.AI+OR+cat:cs.LG" in url
    assert "max_results=10" in url


def test_arxiv_client_url_combined():
    client = ArxivClient(query="diffusion models", categories=["cs.CV"], sort_by="submittedDate")
    url = client.build_query_url()
    assert "+AND+" in url
    assert "sortBy=submittedDate" in url


def test_arxiv_client_fetch_papers():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.content = SAMPLE_ARXIV_XML.encode("utf-8")
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        client = ArxivClient(query="attention is all you need")
        papers = client.fetch_papers()

        assert len(papers) == 1
        p = papers[0]
        assert p.arxiv_id == "1706.03762v7"
        assert p.title == "Attention Is All You Need"
        assert "Ashish Vaswani" in p.authors
        assert p.primary_category == "cs.CL"
        assert "cs.AI" in p.categories
        assert str(p.pdf_url) == "http://arxiv.org/pdf/1706.03762v7"


# ── CLI Commands Help & Integration ──────────────────────────────────────────

def test_cli_arxiv_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["arxiv", "--help"])
    assert result.exit_code == 0
    assert "Search arXiv and interactively select papers" in result.output
    assert "--download-all" in result.output
    assert "--select" in result.output


def test_cli_discover_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["discover", "--help"])
    assert result.exit_code == 0
    assert "--download-all" in result.output


def test_cli_search_arxiv_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["search", "--help"])
    assert result.exit_code == 0
    assert "--arxiv" in result.output


def test_cli_ingest_query_flag():
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "--query" in result.output


def test_cli_arxiv_non_interactive():
    runner = CliRunner()
    with patch("papertrail.ingestion.arxiv_client.ArxivClient.fetch_papers") as mock_fetch:
        mock_fetch.return_value = [
            Paper(
                arxiv_id="1706.03762",
                title="Attention Is All You Need",
                abstract="Summary text",
                authors=["Ashish Vaswani"],
                primary_category="cs.CL",
                categories=["cs.CL"],
                published="2017-06-12T00:00:00Z",
                updated="2017-06-12T00:00:00Z",
            )
        ]
        with patch("papertrail.ingestion.pipeline.index_single_paper") as mock_index:
            mock_index.return_value = (True, "indexed (10 chunks)", 10)

            # Test --select 1
            result = runner.invoke(cli, ["arxiv", "transformer", "--select", "1"])
            assert result.exit_code == 0
            assert "Attention Is All You Need" in result.output
            assert "Indexed 1 new paper(s)" in result.output
            mock_index.assert_called_once()
