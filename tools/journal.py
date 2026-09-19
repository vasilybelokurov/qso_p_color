#!/usr/bin/env python
"""Append a dated entry to JOURNAL.md, capturing the project state automatically.

The point of the journal is that a result can be traced back to the code and the
data that produced it.  Typing that by hand is where it breaks down, so this
script captures the mechanical parts — commit, dirty files, test outcome,
environment — and asks the caller only for the things a machine cannot know:
what was done, what was found, and what is next.

The journal is deliberately **not** committed (see ``.gitignore``).  It records
machine state, timings, dead ends and half-finished thoughts, which are noise in
a shared history but are exactly what you want when returning to the project
three weeks later.

Usage
-----
    python tools/journal.py hook-install     # once: write an entry per commit

    python tools/journal.py add \\
        --what "Fitted the DR9 quasar model" \\
        --found "Held-out log density peaks at K=12" \\
        --next "Background model on the same footprint"

    python tools/journal.py add --what "..." --no-tests   # skip the pytest run

    python tools/journal.py commit           # what the hook calls

The hook covers the mechanical cadence: every commit gets an entry with its
subject, body, diffstat and test state.  It does not cover *findings* — a number
that came out of a fit, a choice settled by a measurement, a hypothesis
abandoned.  Those still need an explicit ``add``, and are the reason the journal
is worth keeping.

Every entry records whether the tests passed *at that moment*.  An entry that
says "tests: not run" is a claim with no evidence behind it, and reads that way
on purpose.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "JOURNAL.md"

HEADER = """# Project journal

Reverse-chronological log of what was done, what came out of it, and what is
next.  Entries below the marker are appended by `tools/journal.py`; the
mechanical fields (commit, test status, environment) are captured at write time,
not typed.

<!-- entries are inserted directly below this marker, newest first -->
<!--JOURNAL-ENTRIES-->
"""


def _run(cmd: list[str], cwd: Path = ROOT) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=1800
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)


def git_state() -> dict[str, str]:
    if not shutil.which("git") or not (ROOT / ".git").exists():
        return {"commit": "not a git repository", "dirty": "", "branch": ""}
    code, commit = _run(["git", "rev-parse", "--short", "HEAD"])
    if code != 0:
        commit = "no commits yet"
    code, branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if code != 0:
        branch = "-"
    _, status = _run(["git", "status", "--porcelain"])
    dirty = ", ".join(sorted({ln[3:].split()[0] for ln in status.splitlines()}))
    return {
        "commit": commit or "no commits yet",
        "branch": branch or "-",
        "dirty": dirty,
    }


def last_commit() -> dict[str, str]:
    """Subject, body, diffstat and changed files of HEAD.

    Uses ``git show`` rather than ``git diff HEAD~1 HEAD`` throughout, because a
    root commit has no parent and the ``HEAD~1`` form fails on it — which is
    exactly the first commit anyone makes, so the failure would go unnoticed
    until it had already written a broken entry.
    """
    _, subject = _run(["git", "log", "-1", "--pretty=%s"])
    _, body = _run(["git", "log", "-1", "--pretty=%b"])

    code, files = _run(["git", "show", "--pretty=format:", "--name-only", "HEAD"])
    names = [f for f in files.splitlines() if f.strip()] if code == 0 else []

    code, stat = _run(["git", "show", "--pretty=format:", "--shortstat", "HEAD"])
    lines = [ln.strip() for ln in stat.splitlines() if "changed" in ln] if code == 0 else []

    return {
        "subject": subject,
        "body": body,
        "files": names,
        "stat": lines[0] if lines else "",
    }


def touches_code(files: list[str]) -> bool:
    """True if the commit changed anything the test suite covers."""
    return any(f.startswith(("src/", "tests/", "scripts/", "tools/")) for f in files)


def test_state(run_tests: bool) -> str:
    if not run_tests:
        return "not run"
    code, out = _run([sys.executable, "-m", "pytest", "-q", "--no-header"])
    tail = [ln for ln in out.splitlines() if ln.strip()]
    summary = tail[-1] if tail else "no output"
    return f"{'PASS' if code == 0 else 'FAIL'} ({summary})"


def env_state() -> str:
    try:
        import numpy

        np_v = numpy.__version__
    except Exception:  # noqa: BLE001
        np_v = "?"
    return f"python {platform.python_version()}, numpy {np_v}, {platform.system()}"


def add_entry(args: argparse.Namespace) -> None:
    if not JOURNAL.exists():
        JOURNAL.write_text(HEADER)

    text = JOURNAL.read_text()
    marker = "<!--JOURNAL-ENTRIES-->"
    if marker not in text:
        text = text.rstrip() + "\n\n" + marker + "\n"

    g = git_state()
    stamp = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    lines = [f"\n## {stamp} — {args.what}\n"]
    if args.found:
        lines.append(f"**Found.** {args.found}\n")
    if args.next:
        lines.append(f"**Next.** {args.next}\n")
    if args.files:
        lines.append(f"**Files.** {', '.join(args.files)}\n")
    lines.append(
        f"**State.** commit `{g['commit']}` on `{g['branch']}`"
        + (f"; uncommitted: {g['dirty']}" if g["dirty"] else "; working tree clean")
        + f". Tests: {test_state(not args.no_tests)}. Env: {env_state()}.\n"
    )

    entry = "\n".join(lines)
    JOURNAL.write_text(text.replace(marker, marker + entry, 1))
    print(f"appended entry to {JOURNAL.relative_to(ROOT)}")


def commit_entry(args: argparse.Namespace) -> None:
    """Append an entry describing HEAD.  This is what the post-commit hook runs.

    Tests run only when the commit touched code, so a docs-only commit does not
    pay 25 seconds — and does not claim a test result it did not earn.
    """
    c = last_commit()
    run_tests = touches_code(c["files"]) and not args.no_tests

    found = c["body"].strip()
    detail = []
    if c["stat"]:
        detail.append(c["stat"])
    if c["files"]:
        shown = ", ".join(c["files"][:12])
        if len(c["files"]) > 12:
            shown += f", and {len(c['files']) - 12} more"
        detail.append(shown)
    if detail:
        found = (found + "\n\n" if found else "") + "**Changed.** " + " — ".join(detail)

    ns = argparse.Namespace(
        what=c["subject"] or "(no commit subject)",
        found=found,
        next="",
        files=[],
        no_tests=not run_tests,
        quiet=args.quiet,
    )
    add_entry(ns)


def hook_install(_: argparse.Namespace) -> None:
    """Install a post-commit hook that journals every commit automatically."""
    hooks = ROOT / ".git" / "hooks"
    if not hooks.exists():
        sys.exit("no .git/hooks directory; run git init first")
    hook = hooks / "post-commit"
    hook.write_text(
        "#!/bin/sh\n"
        "# Appends a JOURNAL.md entry for each commit (installed by tools/journal.py).\n"
        "# JOURNAL.md is gitignored, so this never dirties the working tree.\n"
        "# Set QSO_JOURNAL_SKIP=1 to bypass, QSO_JOURNAL_NO_TESTS=1 to skip pytest.\n"
        '[ -n "$QSO_JOURNAL_SKIP" ] && exit 0\n'
        'flags=""\n'
        '[ -n "$QSO_JOURNAL_NO_TESTS" ] && flags="--no-tests"\n'
        f'"{sys.executable}" "{ROOT}/tools/journal.py" commit $flags --quiet || true\n'
    )
    hook.chmod(0o755)
    print(f"installed {hook}")
    print("  every commit now appends an entry; findings still need an explicit "
          "`journal.py add`")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(required=True)

    a = sub.add_parser("add", help="append an entry")
    a.add_argument("--what", required=True, help="what was done, one line")
    a.add_argument("--found", default="", help="what the result was")
    a.add_argument("--next", default="", help="the next concrete step")
    a.add_argument("--files", nargs="*", default=[], help="files touched")
    a.add_argument("--no-tests", action="store_true", help="do not run pytest")
    a.add_argument("--quiet", action="store_true")
    a.set_defaults(func=add_entry)

    c = sub.add_parser("commit", help="append an entry describing HEAD")
    c.add_argument("--no-tests", action="store_true")
    c.add_argument("--quiet", action="store_true")
    c.set_defaults(func=commit_entry)

    h = sub.add_parser("hook-install", help="append an entry on every git commit")
    h.set_defaults(func=hook_install)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
