from __future__ import annotations

import gzip
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from integrations.search.arxiv import parse_arxiv_atom
from integrations.search.dispatch import format_search_results, search_web, source_names_for_tool
from integrations.search.github import parse_github_repos
from integrations.search.openalex import parse_openalex_works
from integrations.search.registry import names
from integrations.search.types import SearchHit
from integrations.search.wikipedia import WikipediaSource, parse_wikipedia_opensearch


_EXPECTED_SOURCES = frozenset(
    {
        "arxiv",
        "crossref",
        "cse",
        "ddg",
        "europepmc",
        "github",
        "hn",
        "huggingface",
        "mdn",
        "openalex",
        "pubmed",
        "searxng",
        "semanticscholar",
        "stackexchange",
        "wikidata",
        "wikipedia",
        "zenodo",
    }
)

_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678</id>
    <title>Sound Event Detection</title>
    <summary>A paper about SED.</summary>
    <published>2024-03-01T00:00:00Z</published>
    <author><name>Jane Doe</name></author>
  </entry>
</feed>
"""

_WIKI_OPENSEARCH = [
    "sound event detection",
    ["Sound event detection"],
    ["Detection of acoustic events in audio."],
    ["https://en.wikipedia.org/wiki/Sound_event_detection"],
]

_OPENALEX_WORK = {
    "id": "https://openalex.org/W123",
    "title": "Open SED",
    "doi": "https://doi.org/10.1/sed",
    "publication_year": 2024,
    "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
    "abstract_inverted_index": {"Hello": [0], "world": [1]},
}

_GITHUB_PAYLOAD = {
    "items": [
        {
            "full_name": "foo/sed",
            "html_url": "https://github.com/foo/sed",
            "description": "SED toolkit",
            "license": {"spdx_id": "MIT"},
        }
    ]
}


class FakeResponse:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class SearchRegistryTests(unittest.TestCase):
    def test_registry_contains_documented_sources(self) -> None:
        registered = set(names())
        self.assertEqual(registered, set(_EXPECTED_SOURCES))

    def test_unknown_source_lists_legal_names(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            search_web("query", source="nope")
        message = str(ctx.exception)
        self.assertIn("Unknown search source", message)
        self.assertIn("auto", message)
        self.assertIn("wikipedia", message)
        self.assertIn("arxiv", message)

    def test_auto_does_not_call_a_vertical(self) -> None:
        mapped = [{"title": "Hit", "url": "https://example.com/a", "snippet": "s"}]
        with patch(
            "integrations.google.searxng.searxng_configured",
            return_value=True,
        ), patch(
            "integrations.google.searxng.fetch_searxng_hits",
            return_value=mapped,
        ), patch(
            "integrations.search.wikipedia.WikipediaSource.search"
        ) as wiki, patch(
            "integrations.search.arxiv.ArxivSource.search"
        ) as arxiv, patch(
            "integrations.search.github.GithubSource.search"
        ) as github, patch(
            "integrations.search.cse.CseSource.search"
        ) as cse:
            hits = search_web("sed", source="auto")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].url, "https://example.com/a")
        wiki.assert_not_called()
        arxiv.assert_not_called()
        github.assert_not_called()
        cse.assert_not_called()


class SearchParseTests(unittest.TestCase):
    def test_wikipedia_opensearch_maps_title_url_snippet(self) -> None:
        hits = parse_wikipedia_opensearch(_WIKI_OPENSEARCH, limit=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "Sound event detection")
        self.assertEqual(hits[0].url, "https://en.wikipedia.org/wiki/Sound_event_detection")
        self.assertEqual(hits[0].snippet, "Detection of acoustic events in audio.")
        self.assertEqual(hits[0].extras.get("license"), "CC BY-SA")

    def test_arxiv_atom_maps_title_url_snippet(self) -> None:
        hits = parse_arxiv_atom(_ARXIV_ATOM, limit=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "Sound Event Detection")
        self.assertEqual(hits[0].url, "http://arxiv.org/abs/1234.5678")
        self.assertEqual(hits[0].snippet, "A paper about SED.")
        self.assertEqual(hits[0].extras.get("year"), "2024")
        self.assertEqual(hits[0].extras.get("authors"), "Jane Doe")
        self.assertEqual(hits[0].extras.get("id"), "1234.5678")

    def test_openalex_json_maps_title_url_snippet(self) -> None:
        hits = parse_openalex_works([_OPENALEX_WORK], limit=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "Open SED")
        self.assertEqual(hits[0].url, "https://doi.org/10.1/sed")
        self.assertEqual(hits[0].snippet, "Hello world")
        self.assertEqual(hits[0].extras.get("year"), "2024")
        self.assertEqual(hits[0].extras.get("authors"), "Ada Lovelace")

    def test_github_json_maps_title_url_snippet(self) -> None:
        hits = parse_github_repos(_GITHUB_PAYLOAD, limit=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "foo/sed")
        self.assertEqual(hits[0].url, "https://github.com/foo/sed")
        self.assertEqual(hits[0].snippet, "SED toolkit")
        self.assertEqual(hits[0].extras.get("license"), "MIT")

    def test_github_query_drops_filler_tokens(self) -> None:
        from integrations.search.arxiv import arxiv_search_query
        from integrations.search.github import prepare_github_query

        self.assertEqual(
            prepare_github_query("recent Sound Event Detection datasets 2024 2025 github"),
            "Sound Event Detection datasets",
        )
        self.assertEqual(prepare_github_query("DCASE 2024"), "DCASE 2024")
        self.assertEqual(prepare_github_query("RealDESED"), "RealDESED")
        self.assertEqual(arxiv_search_query("RealDESED"), "all:RealDESED")
        self.assertEqual(arxiv_search_query("Sound Event Detection"), 'all:"Sound Event Detection"')

    def test_zenodo_json_maps_title_url_snippet(self) -> None:
        from integrations.search.zenodo import parse_zenodo_records

        payload = {
            "hits": {
                "hits": [
                    {
                        "doi": "10.5281/zenodo.20056072",
                        "links": {"html": "https://zenodo.org/records/20056072"},
                        "metadata": {
                            "title": "RealDESED",
                            "description": "<p>A real-world benchmark.</p>",
                            "publication_date": "2026-01-01",
                            "creators": [{"name": "Schmid, Florian"}],
                        },
                    }
                ]
            }
        }
        hits = parse_zenodo_records(payload, limit=5)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "RealDESED")
        self.assertEqual(hits[0].url, "https://zenodo.org/records/20056072")
        self.assertEqual(hits[0].snippet, "A real-world benchmark.")
        self.assertEqual(hits[0].extras.get("doi"), "10.5281/zenodo.20056072")
        self.assertEqual(hits[0].extras.get("year"), "2026")

    def test_format_search_results_includes_extras(self) -> None:
        hit = SearchHit(
            title="T",
            url="https://example.com",
            snippet="S",
            extras={"year": "2024", "doi": "10.1/x"},
        )
        payload = format_search_results([hit], query="q")
        self.assertEqual(payload["query"], "q")
        self.assertEqual(payload["results"][0]["year"], "2024")
        self.assertEqual(payload["results"][0]["doi"], "10.1/x")

    def test_wikipedia_search_uses_mocked_urlopen(self) -> None:
        extract = {
            "query": {
                "pages": {
                    "1": {"extract": "Long extract about SED."}
                }
            }
        }

        def fake_urlopen(request: Request, timeout: object = None) -> FakeResponse:
            url = request.full_url
            if "action=opensearch" in url:
                body = json.dumps(_WIKI_OPENSEARCH).encode("utf-8")
                return FakeResponse(body)
            if "prop=extracts" in url:
                return FakeResponse(json.dumps(extract).encode("utf-8"))
            raise AssertionError(url)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            hits = WikipediaSource().search("sed", 3)
        self.assertEqual(hits[0].snippet, "Long extract about SED.")
        self.assertEqual(hits[0].url, "https://en.wikipedia.org/wiki/Sound_event_detection")


class SearchHttpTests(unittest.TestCase):
    def test_gzip_body_is_decoded(self) -> None:
        from integrations.search.http import fetch_json

        payload = {"ok": True}
        raw = gzip.compress(json.dumps(payload).encode("utf-8"))

        def fake_urlopen(request: Request, timeout: object = None) -> FakeResponse:
            return FakeResponse(raw, headers={"Content-Encoding": "gzip"})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            data = fetch_json("https://example.com/x", source="stackexchange")
        self.assertEqual(data, payload)


class SearchCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._disabled = os.environ.get("SOPHON_SEARCH_DISABLED")
        self._tools = os.environ.get("SOPHON_WEB_SEARCH_TOOLS")
        self._contact = os.environ.get("SOPHON_SEARCH_CONTACT")
        os.environ.pop("SOPHON_SEARCH_DISABLED", None)

    def tearDown(self) -> None:
        self._restore("SOPHON_SEARCH_DISABLED", self._disabled)
        self._restore("SOPHON_WEB_SEARCH_TOOLS", self._tools)
        self._restore("SOPHON_SEARCH_CONTACT", self._contact)

    def _restore(self, key: str, value: str | None) -> None:
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    def test_status_lists_auto_and_registry(self) -> None:
        from integrations.search.status import handle_search_command

        text = handle_search_command("")
        self.assertIn("wikipedia", text)
        self.assertIn("arxiv", text)
        self.assertIn("zenodo", text)
        self.assertIn("github", text)
        self.assertIn("auto", text)
        self.assertIn("searxng", text)
        self.assertIn("cse", text)

    def test_disable_omits_source_from_tool_enum(self) -> None:
        from integrations.search.status import handle_search_command

        handle_search_command("disable wikipedia")
        self.assertNotIn("wikipedia", source_names_for_tool())
        with self.assertRaises(RuntimeError) as ctx:
            search_web("sed", source="wikipedia")
        self.assertIn("disabled", str(ctx.exception))
        handle_search_command("enable all")
        self.assertIn("wikipedia", source_names_for_tool())

    def test_on_off_and_contact(self) -> None:
        from integrations.search.dispatch import web_search_tools_enabled
        from integrations.search.http import search_contact
        from integrations.search.status import handle_search_command

        self.assertEqual(handle_search_command("off"), "web_search tool: off")
        self.assertFalse(web_search_tools_enabled())
        self.assertEqual(handle_search_command("on"), "web_search tool: on")
        self.assertTrue(web_search_tools_enabled())
        handle_search_command("contact you@example.com")
        self.assertEqual(search_contact(), "you@example.com")
        handle_search_command("contact clear")
        self.assertEqual(search_contact(), "")

    def test_disable_auto_is_rejected(self) -> None:
        from integrations.search.status import handle_search_command

        with self.assertRaises(ValueError):
            handle_search_command("disable auto")


if __name__ == "__main__":
    unittest.main()
