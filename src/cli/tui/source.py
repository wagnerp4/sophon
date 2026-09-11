from __future__ import annotations

from pathlib import Path

from textual.widgets import TextArea

from processing.text.completion import (
    CompletionQuery,
    PrefixCompleter,
    extract_prefix_at_cursor,
)


class EditorSourceTextArea(TextArea):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._completer: PrefixCompleter | None = None
        self._file_path: Path | None = None
        self._language: str | None = None

    def bind_completer(self, completer: PrefixCompleter | None) -> None:
        self._completer = completer

    def set_completion_context(self, path: Path | None, language: str | None) -> None:
        self._file_path = path
        self._language = language

    def update_suggestion(self) -> None:
        if self.read_only or self._completer is None:
            self.suggestion = ""
            return
        text = self.text
        try:
            offset = self.document.get_index_from_location(self.cursor_location)
        except Exception:
            self.suggestion = ""
            return
        prefix, suffix = extract_prefix_at_cursor(text, offset)
        if not prefix:
            self.suggestion = ""
            return
        candidate = self._completer.complete(
            CompletionQuery(
                prefix=prefix,
                suffix=suffix,
                document_text=text,
                path=self._file_path,
                language=self._language,
            )
        )
        self.suggestion = candidate.text if candidate is not None else ""
