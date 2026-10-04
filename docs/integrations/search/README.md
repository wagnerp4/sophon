# Search (`web_search`)

One chat tool. Backends are `source` values on that tool. Broad web is SearXNG, then Google Custom Search. Named values call one no-key vertical. `source=auto` does not fan out to Wikipedia, arXiv, GitHub, or the rest.

`google off` in the HUD is Gmail/Drive. `web_search` is a separate tool. It does not drive a browser.

## Env

```powershell
$env:SOPHON_WEB_SEARCH_TOOLS="1"
$env:SOPHON_SEARCH_CONTACT="you@example.com"
$env:SOPHON_SEARXNG_URL="http://127.0.0.1:8888"
```

| Variable | Role |
| --- | --- |
| `SOPHON_WEB_SEARCH_TOOLS` | Set to `0` / `false` / `no` / `off` to hide `web_search`. Any other value (including unset) lists the tool. Verticals need no key. |
| `SOPHON_SEARCH_CONTACT` | Polite-pool mailto for Crossref, OpenAlex, PubMed (`tool=sophon`). Empty still calls the APIs. |
| `SOPHON_SEARCH_DISABLED` | Comma list of source names hidden from `web_search`. `/search disable NAME` sets this for the process. |
| `SOPHON_SEARXNG_URL` | Preferred `auto` backend. JSON must be enabled on the instance. |
| `SOPHON_GOOGLE_CSE_KEY` / `SOPHON_GOOGLE_CSE_CX` | Fallback `auto` backend. See [../google/README.md](../google/README.md). |

User-Agent is `Sophon/0.1 (search; CONTACT; +https://github.com/wagnerp4/sophon)` when contact is set.

## `source` values

| `source` | Use when |
| --- | --- |
| `auto` | Broad web. SearXNG if it returns hits, else CSE. Errors if neither meta backend is configured. |
| `searxng` | Self-hosted SearXNG only. |
| `cse` | Google Custom Search only. |
| `wikipedia` | Encyclopedia titles and a short extract. License extra `CC BY-SA`. |
| `wikidata` | Entity keyword search (`wbsearchentities`). Not SPARQL. |
| `ddg` | DuckDuckGo Instant Answer JSON. Abstracts and related topics. Often empty. Not a web SERP. |
| `arxiv` | Atom API. 3s minimum interval. Sends `Accept-Encoding: gzip` (urllib `identity` often yields HTTP 406). |
| `crossref` | Works query. |
| `openalex` | Works search. |
| `pubmed` | E-utilities esearch then esummary. |
| `europepmc` | Europe PMC REST. |
| `semanticscholar` | Graph `paper/search`. Unauthenticated pool. |
| `zenodo` | Open research records and datasets (`mostrecent`). |
| `github` | Repository search, sorted by `updated`. Filler tokens (`latest`, years, the word `github`) are stripped. |
| `huggingface` | Public model and dataset search. |
| `stackexchange` | Stack Overflow advanced search. No key (300/day). |
| `hn` | Hacker News Algolia. |
| `mdn` | MDN document search. |

Hits are `title`, `url`, `snippet`, plus `extras` when present (`year`, `authors`, `doi`, `id`, `license`).

## Slash command

```text
/search
/search on
/search off
/search contact you@example.com
/search disable github
/search enable github
/search enable all
/search wikipedia
/search wikipedia sound event detection
```

`/search` lists every `source` with family and state (`ready` / `missing` / `disabled`). `/search SOURCE` prints one backend. `/search SOURCE QUERY` runs that backend (limit 5). `/search-status` is an alias.

`on` / `off` / `contact` / `enable` / `disable` write process env (`SOPHON_WEB_SEARCH_TOOLS`, `SOPHON_SEARCH_CONTACT`, `SOPHON_SEARCH_DISABLED`). They are not written back to `.env`. Restarting chat reloads the file.

## Out of this slice

Brave, Mojeek, Mwmbl, Marginalia, YaCy, HTML scrapers (`ddgs`), OSM, a second tool name.

**Fetch/read page body** is not a `web_search` backend. Next direction uses one allowlisted scrape MCP over stdio ([../../direction/next-mcp-stdio-scrape.md](../../direction/next-mcp-stdio-scrape.md)). Native `/search` stays.
