"""Build the static GitHub Pages site from the engine's Markdown reports (no dependencies).

    python scripts/build_site.py store/reports site
"""

from __future__ import annotations

import datetime as dt
import html
import re
import sys
from pathlib import Path

PAGES = [("RESEARCH_REPORT.md", "Research report (pre-registered experiments)"),
         ("BACKTEST_REPORT.md", "Walk-forward backtest of the daily model")]

CSS = """body{font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
main{max-width:1100px;margin:auto;padding:24px}a{color:#8cc8ff}table{border-collapse:collapse;margin:12px 0;font-size:13px}
td,th{border:1px solid #333;padding:4px 8px;vertical-align:top}th{background:#1b1f27}code,pre{background:#1b1f27;padding:2px 4px}
pre{padding:10px;overflow:auto}h1,h2,h3{line-height:1.25}.muted{color:#9aa}"""


def inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"_(.+?)_", r"<i>\1</i>", s)
    return s


def md_to_html(md: str) -> str:
    out, lines, i = [], md.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1:j])) + "</pre>")
            i = j + 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr>" +
                           "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:]) + "</table>")
            continue
        m = re.match(r"^(#{1,4}) (.*)", ln)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
        elif ln.startswith(("* ", "- ")):
            items = []
            while i < len(lines) and lines[i].startswith(("* ", "- ")):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return "\n".join(out)


def page(title: str, body: str) -> str:
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{body}</main></body></html>")


def main(src: str, dst: str) -> None:
    s, d = Path(src), Path(dst)
    d.mkdir(parents=True, exist_ok=True)
    links = []
    for fn, title in PAGES:
        f = s / fn
        if not f.exists():
            continue
        name = fn.replace(".md", ".html").lower()
        (d / name).write_text(page(title, "<p><a href='index.html'>&larr; index</a></p>" + md_to_html(f.read_text(encoding="utf-8"))),
                              encoding="utf-8")
        links.append(f"<li><a href='{name}'>{html.escape(title)}</a></li>")
    body = ("<h1>NIFTY engine v5 &mdash; daily reports</h1>"
            f"<p class=muted>Updated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC by GitHub Actions. Paper trading and research "
            "only; nothing here is investment advice. Experiments that need Upstox data or the local LLM run on the "
            "owner's machine and show as INSUFFICIENT_DATA here.</p><ul>" + "".join(links) + "</ul>")
    (d / "index.html").write_text(page("NIFTY engine v5", body), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "store/reports", sys.argv[2] if len(sys.argv) > 2 else "site")
