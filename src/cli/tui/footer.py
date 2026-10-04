from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Label

from cli.tui.speech import SpeakButton


class FooterCopy(Label):
    DEFAULT_CSS = """
    FooterCopy {
        width: auto;
        height: 1;
        min-height: 1;
        padding: 0 1;
        color: $footer-foreground;
        background: $footer-background;
    }
    """

    def __init__(self) -> None:
        super().__init__("Copy", id="chat-copy")

    def on_click(self, event) -> None:
        event.stop()
        handler = getattr(self.screen, "action_copy_chat", None)
        if callable(handler):
            handler()


class NexusFooter(Footer):
    DEFAULT_CSS = """
    NexusFooter {
        height: 1;
        dock: bottom;
    }

    NexusFooter #footer-right {
        dock: right;
        width: auto;
        height: 1;
        layout: horizontal;
    }

    NexusFooter Label.-panes {
        width: auto;
        min-width: 16;
        height: 1;
        padding: 0 1;
        color: $footer-description-foreground;
        background: $footer-description-background;
    }

    NexusFooter FooterCopy,
    NexusFooter SpeakButton {
        width: auto;
        height: 1;
        min-height: 1;
        max-height: 1;
        padding: 0 1;
        background: $footer-background;
        color: $footer-foreground;
    }

    NexusFooter SpeakButton {
        min-width: 8;
        border: none;
    }
    """

    def __init__(self, *args, chat_actions: bool = False, **kwargs) -> None:
        self._chat_actions = chat_actions
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        from cli.tui.keybinds import panes_footer_label

        yield from super().compose()
        with Horizontal(id="footer-right"):
            if self._chat_actions:
                yield FooterCopy()
                yield SpeakButton("chat-speak")
            yield Label(panes_footer_label(), classes="-panes")

    def set_panes_label(self, text: str) -> None:
        try:
            label = self.query_one("Label.-panes", Label)
        except Exception:
            return
        label.update(text)
