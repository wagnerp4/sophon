from __future__ import annotations

import sys
from pathlib import Path

_src_root = Path(__file__).resolve().parent.parent
_src_root_s = str(_src_root)
if _src_root_s not in sys.path:
    sys.path.insert(0, _src_root_s)

import click

from utils.device.env_bootstrap import load_sophon_dotenv


@click.command("rag-index", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--vault",
    "vault_root",
    default=None,
    help="Obsidian vault filesystem root. Env: SOPHON_VAULT_PATH.",
)
@click.option(
    "--project/--no-project",
    "include_project",
    default=True,
    show_default=True,
    help="Include the Sophon project tree in the corpus.",
)
@click.option(
    "--project-root",
    default=None,
    help="Override project root (default: package project root).",
)
@click.option(
    "--index",
    "index_path",
    default=None,
    help="LEANN index path (default: data/rag/indexes/default or SOPHON_LEANN_INDEX).",
)
@click.option(
    "--structure-dir",
    default=None,
    help="Optional LightRAG working directory (env: SOPHON_LIGHTRAG_DIR).",
)
@click.option("--rebuild", is_flag=True, default=False, help="Wipe existing LEANN index files first.")
@click.option("--chunk-chars", default=1200, type=int, show_default=True)
@click.option("--overlap", default=150, type=int, show_default=True)
@click.option("--embedding-model", default=None, help="Optional LEANN embedding model override.")
def rag_index_command(
    vault_root: str | None,
    include_project: bool,
    project_root: str | None,
    index_path: str | None,
    structure_dir: str | None,
    rebuild: bool,
    chunk_chars: int,
    overlap: int,
    embedding_model: str | None,
) -> None:
    """Build the default vault+project LEANN corpus (optional LightRAG insert)."""
    load_sophon_dotenv()
    from processing.text.retrieval.corpus import build_default_corpus

    def on_progress(msg: str) -> None:
        click.echo(f"[sophon] {msg}", err=True)

    try:
        result = build_default_corpus(
            vault_root=vault_root,
            project_root=project_root,
            include_project=include_project,
            index_path=index_path,
            structure_dir=structure_dir,
            rebuild=rebuild,
            chunk_chars=chunk_chars,
            overlap=overlap,
            embedding_model=embedding_model,
            on_progress=on_progress,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(
        f"index={result.index_path} files={result.n_files} chunks={result.n_chunks} "
        f"vault={result.vault_root or '-'} project={result.project_root or '-'}"
    )
    for line in result.messages:
        click.echo(line)


def main() -> None:
    rag_index_command()


if __name__ == "__main__":
    main()
