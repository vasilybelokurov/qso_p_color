# qso_p_color

Read **`AGENTS.md`** first — it is the working contract for this repository
(the question, the non-negotiable rules, the work order, the output contract).

`docs/REVIEW_OF_PLAN.md` records why the implementation departs from
`qso_binary_color_probability_plan.md` where it does, with the measurements
behind each departure.

Quick start:

```bash
source ~/Work/venvs/.venv/bin/activate
cd ~/Work/Code/qso_p_color
pip install -e . --no-deps
python -m pytest -q
```

`JOURNAL.md` is local and gitignored. Install the hook once per clone so every
commit is journalled automatically:

```bash
python tools/journal.py hook-install
```

The hook records what changed; it cannot record what you *learned*. Add that
yourself after any substantive step:

```bash
python tools/journal.py add --what "..." --found "..." --next "..."
```
