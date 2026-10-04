from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from cli.tui.choice import ChoicePrompt


class JobQueueScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("a", "apply_company", "All at company"),
    ]

    DEFAULT_CSS = """
    JobQueueScreen {
        align: center middle;
        background: $background 60%;
    }

    #job-dialog {
        width: 88;
        max-width: 96%;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #job-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #job-queue {
        height: auto;
        max-height: 16;
    }

    #job-hint {
        margin-top: 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict[str, object]] = []
        self._open_count = 0
        self._filling = False

    def compose(self) -> ComposeResult:
        with Vertical(id="job-dialog"):
            yield Static("jobs", id="job-title")
            yield ListView(id="job-queue")
            yield Static("", id="job-hint")

    async def on_mount(self) -> None:
        await self._reload(write_digest=True)
        self.query_one("#job-queue", ListView).focus()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if self._filling:
            return
        row = self._row_at(event.list_view.index)
        if row is None:
            return
        self._ask(row)

    def action_apply_company(self) -> None:
        panel = self.query_one("#job-queue", ListView)
        row = self._row_at(panel.index)
        if row is None:
            return
        from integrations.jobs.feeds import set_company_applied
        from integrations.jobs.match import match_jobs
        from integrations.jobs.store import VaultStore

        set_company_applied(VaultStore(), str(row.get("slug") or ""))
        match_jobs(VaultStore())
        self.run_worker(self._reload(write_digest=False), exclusive=True)

    def action_close(self) -> None:
        self.dismiss(None)

    def _ask(self, row: dict[str, object]) -> None:
        title = str(row.get("legal_name") or "") + " — " + str(row.get("role") or "")

        def _done(choice: str | None) -> None:
            if choice not in ("applied", "skip"):
                return
            from integrations.jobs.feeds import set_role_status
            from integrations.jobs.match import match_jobs
            from integrations.jobs.store import VaultStore

            store = VaultStore()
            set_role_status(
                store,
                str(row.get("slug") or ""),
                str(row.get("url") or ""),
                choice,
                str(row.get("role") or ""),
            )
            match_jobs(store)
            self.run_worker(self._reload(write_digest=False), exclusive=True)

        self.app.push_screen(
            ChoicePrompt(
                title,
                [
                    ("applied", "applied", True),
                    ("skip", "skip", True),
                ],
            ),
            _done,
        )

    async def _reload(self, *, write_digest: bool) -> None:
        from integrations.jobs.match import match_jobs, queue_top
        from integrations.jobs.store import VaultStore

        store = VaultStore()
        if write_digest:
            match_jobs(store)
        rows, count = queue_top(store)
        self._rows = rows
        self._open_count = count
        self._filling = True
        panel = self.query_one("#job-queue", ListView)
        await panel.clear()
        for index, row in enumerate(rows, start=1):
            label = (
                str(index)
                + ". "
                + str(row.get("legal_name") or "")
                + "  "
                + str(row.get("score") or "")
                + "  "
                + str(row.get("role") or "")
            )
            panel.append(ListItem(Label(label)))
        self._filling = False
        hint = "Enter or click: applied or skip. a: all open roles at this company. Esc closes."
        if count < 10:
            hint = "open roles: " + str(count) + ". /jobs discover. " + hint
        self.query_one("#job-hint", Static).update(hint)
        self.query_one("#job-title", Static).update("jobs  track " + _track(store))

    def _row_at(self, index: int | None) -> dict[str, object] | None:
        if index is None or index < 0 or index >= len(self._rows):
            return None
        return self._rows[index]


def _track(store: object) -> str:
    from integrations.jobs.criterion import load_criterion

    return str(load_criterion(store).get("track") or "student")
