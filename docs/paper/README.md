# AgriBot — IEEE paper draft

Working directory for the IEEE manuscript.

## Files

| File | Purpose |
|---|---|
| `main.tex` | IEEEtran journal-mode skeleton with all section stubs and figure includes |
| `bibliography.bib` | 35 references (Tier 1 foundational + Tier 2/3 framing + PaperAnalysis comparators) |
| `figures/` *(symlink target)* | Use the rendered figures in `../figures/` — `\includegraphics{figures/fig1_architecture.pdf}` etc. |

## How to compile

### Overleaf (recommended)

1. New Project → Upload → upload `main.tex` + `bibliography.bib`
2. Upload all `docs/figures/fig*.pdf` files into a `figures/` subfolder in your Overleaf project
3. Overleaf auto-detects IEEEtran
4. Compile (Recompile button)

### Local

```bash
cd docs/paper
pdflatex main
bibtex main
pdflatex main
pdflatex main   # third pass to resolve cross-references
```

You'll need the figures in `docs/paper/figures/` (copy or symlink from `docs/figures/`).

## Switching from journal (IEEE Access) to conference (TENCON, etc.)

Change line 1 of `main.tex`:

```latex
\documentclass[journal]{IEEEtran}     % long-form
% becomes
\documentclass[conference]{IEEEtran}  % 6-8 page conference
```

Also comment out the `IEEEbiographynophoto` blocks at the end (conference papers don't include bios).

## Page-length targets

| Venue | Body pages | + references | Notes |
|---|---|---|---|
| IEEE Access (journal) | 10–15 | 1–2 | Long-form, abstract 150–250 words, author bios required |
| IEEE TENCON | 6 | up to 8 with $100/page | Tighter — must trim Methods + Related Work |
| IEEE COMPSAC | 6–10 | varies by track | Research/SETA/STA tracks have different limits |
| ICCIT / regional | 4–6 | included | Trim aggressively |

## Drafting status

Each section has `\todo{...}` and `\fixme{...}` markers. Search for these:

```bash
grep -n "fixme\|todo" main.tex
```

The `\fixme` macro renders red bold in the PDF; `\todo` renders orange bold. Both must be resolved before submission. Delete the macro definitions in the preamble for final.

## Workflow

1. Fill in `\fixme{...}` author block (page 1)
2. Section 3 (Architecture) and 4 (Implementation) first — easiest
3. Section 2 (Related Work) — condense from `../../PaperAnalysis/reviews/*.md`
4. Section 5 (Eval) and 6 (Results) — paste from `../EVALUATION.md` and adapt
5. Section 1 (Intro), 7 (Discussion), 8 (Conclusion) — last
6. Abstract — last, once numbers are locked

## AI assistance disclosure (required by IEEE policy)

The `Acknowledgment` section already has the standard disclosure stub:
> "Draft text and figure scripts were prepared with assistance from a
> large language model coding assistant; all technical decisions,
> experiments, evaluations, and the final manuscript are the authors'
> responsibility."

Keep this in the final version — IEEE requires AI-assistance disclosure since 2023.
