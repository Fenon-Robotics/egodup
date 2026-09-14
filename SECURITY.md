# Security policy

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could expose source videos, local paths, credentials, model-cache integrity, or unsafe media parsing. Report it privately through GitHub's **Security → Report a vulnerability** flow for this repository.

Include the affected version, reproduction steps, impact, and any suggested mitigation. Do not attach private footage; construct a minimal synthetic reproduction when possible.

## Security boundaries

- Videos are untrusted parser inputs. Keep FFmpeg/PyAV and system libraries supported and patched.
- Model files are executable trusted artifacts. `egodup` loads only the approved checksum-pinned model and never a model supplied with a submission.
- Normal processing is local-first and does not upload source media.
- HTML evidence can contain sensitive paths or future thumbnails; store it with appropriate access controls and retention.
- A detection result is not an authorization decision and must not directly trigger deletion, rejection, or payment action.

Only the latest commit on `main` currently receives security fixes while the project remains pre-1.0.
