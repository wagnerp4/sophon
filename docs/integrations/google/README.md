# Google (Gmail, Drive, bookmarks, Custom Search)

sophon can connect Google accounts for Gmail search and Drive browsing, load a local Chrome bookmark export, and run Google Custom Search.

| Surface | How |
|--------|-----|
| **Editor** | View → Drive / Bookmarks trees. File → Search Mail… / Web Search…. View → Google accounts to connect/switch. |
| **Chat** (LM Studio) | Tools: `gmail_search`, `gmail_read`, `drive_tree`, `drive_list`, `bookmarks_tree`, `web_search` |

## Prerequisites

1. Google Cloud project with **Gmail API**, **Drive API**, and **Custom Search API** enabled.
2. OAuth consent screen: External, **Testing**. Add your Gmail addresses as test users.
3. Create an **OAuth Desktop** client. Download the JSON.
4. For web search: create an API key and a [Programmable Search Engine](https://programmablesearchengine.google.com/) with “Search the entire web”. Copy the engine id (`cx`).
5. Export Chrome bookmarks (HTML) into a folder, or point at a single `bookmarks_*.html` file.

TUM / Outlook mail is not covered by the Gmail API.

## Env

Prefer the sophon repo `.env` (loaded via `load_sophon_dotenv`):

```text
SOPHON_GOOGLE_CLIENT_SECRETS=C:\path\to\client_secret.json
SOPHON_GOOGLE_TOOLS=1
SOPHON_BOOKMARKS_PATH=C:\Notes\Obsidian Notes\KB\Data\Private Data\Chrome\Bookmark Exports
SOPHON_GOOGLE_CSE_KEY=...
SOPHON_GOOGLE_CSE_CX=...
SOPHON_WEB_SEARCH_TOOLS=1
```

On WSL, Windows paths in `.env` are resolved to `/mnt/<drive>/...`.

Tokens are stored under `data/google/accounts/<email>.json`. The active account is `data/google/active.json`.

Install Google client libs with the TUI extra:

```powershell
uv sync --extra tui
```

## Editor

1. Restart the TUI after `.env` changes.
2. View → Google accounts → Connect account… (browser / console OAuth).
3. View → Drive tree to browse My Drive (readonly). Selecting a file opens exported text or metadata in a locked editor buffer.
4. View → Bookmarks tree loads the newest `bookmarks_*.html` from `SOPHON_BOOKMARKS_PATH`.
5. File → Search Mail… uses Gmail `q` syntax. File → Web Search… uses Custom Search snippets.

## Chat

With LM Studio and tools enabled, check `/tools`. Ask for mail, Drive folders, bookmarks, or a web query. The model should call the google_* / web_search tools when they are listed.

## Limits

- Readonly Gmail and Drive scopes only.
- Custom Search is not personalized and has a free daily quota (typically 100 queries/day).
- Bookmarks are a static Netscape HTML export, not live Chrome Sync.
- No send-mail, Drive writes, or embedded browser.
