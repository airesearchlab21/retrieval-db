# Anonymous Reviewer Artifact Handoff

This directory records the anonymous upload handoff for the NeurIPS reviewer artifact. It is part of the paper package so the final archive has a stable, reviewer-facing location for the upload identifier, verification steps, and remaining manual gate.

## Anonymous Artifact

- Anonymous artifact ID: `retrieval-db-6FB8`
- Target reviewer URL: `https://anonymous.4open.science/r/retrieval-db-6FB8`
- Local package candidate: the current tarball path is tracked in the local maintainer status file. That status file is intentionally excluded from the tarball to avoid self-referential package hashes.

## Manual Upload Gate

The local package is ready only after the target reviewer URL resolves without authentication. If the URL redirects to a 401 response, create or unlock the 4open artifact from a logged-in GitHub OAuth session, upload the current reviewer package, and rerun the URL check before OpenReview submission.

## Reviewer-Side Verification

After downloading and extracting the reviewer package, run:

```bash
python paper/verify_neurips_artifacts.py --json
```

The expected local verifier result is `pass: true` with 72 archived run directories checked. For a broader executable check, run:

```bash
python -m pytest -q
```

## Anonymity Checks

Before upload, confirm that the package contains no local absolute workspace paths, local user home paths, personal repository URLs, AppleDouble `._*` sidecar files, or author-identifying metadata. The package builder and `paper/verify_neurips_artifacts.py` both perform these checks on the files they stage.
