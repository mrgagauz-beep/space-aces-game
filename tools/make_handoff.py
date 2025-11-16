from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ITER = DOCS / "ITERATION_LOG.md"
SNAP = DOCS / "CONTEXT_SNAPSHOT.md"
OUT = ROOT / "handoff" / "context_clipboard.txt"


def tail_iters(md: str, n: int = 2) -> str:
    blocks = re.split(r"(?=^####\s+Итерация\s+)", md, flags=re.M)
    blocks = [b.strip() for b in blocks if b.strip().startswith("#### Итерация")]
    return "\n\n".join(blocks[-n:]) if blocks else ""


def main() -> None:
    snap_txt = SNAP.read_text(encoding="utf-8")
    iter_txt = ITER.read_text(encoding="utf-8")
    last = tail_iters(iter_txt, 2)

    text = (
        "# Space Aces  Handoff Context\n\n"
        "## Snapshot\n" + snap_txt + "\n\n" +
        ("## Последние итерации\n" + last + "\n\n" if last else "") +
        "## Инструкции разработчику\n"
        "- Микрозадачи; логи+self-test; не ломать интерфейсы.\n"
    )

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text[:12000], encoding="utf-8")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
