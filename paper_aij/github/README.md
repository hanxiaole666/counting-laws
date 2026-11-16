# `paper_aij/github/`

This directory holds the two files that are **not** part of the working tree's
documentation but are needed to publish the artefact:

| file | role |
|---|---|
| `UPLOAD_MANIFEST.md` | what goes into the public repository, what goes into a release, and what is deliberately left out |
| `CITATION.cff` | the citation stub for the public repository (the root `CITATION.cff` is the same thing) |

The repository's front page is `README.md` **at the repository root**, not here.
It is authored there on purpose: a README that lives somewhere else and is copied
at release time is a README that drifts. Every measurement it quotes is inside a
marked block that `tools/refresh_readme.py` regenerates, and
`tests/test_repo_docs.py` fails if a block is stale.
