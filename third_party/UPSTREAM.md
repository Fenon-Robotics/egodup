# Upstream provenance

| Project | Revision | Source paths used | Local changes |
|---|---|---|---|
| facebookresearch/vsc2022 | `644d52118297d8eeaf9ef7483b0246fdf4c8292b` | `vsc/baseline/adapt_sscd_model.py`, `inference_impl.py`, `localization.py`, `sscd_baseline.py` | Reimplemented only checkpoint adaptation, `RESIZE_320_CENTER`, raw/cosine feature contract, and localization boundary adapter. Added real-image parity validation. |
| alipay/VCSL | `29ce63909ce605a73335ce48d1e21f395b4b503c` | `vcsl/vta.py` (`tn`, `iou`) | Vendored only TN and IoU logic. Removed unrelated algorithms/dependencies, replaced removed logger, updated deprecated NumPy boolean use, added deterministic path return. |
| facebookresearch/sscd-copy-detection | `95902662f2217a5f4aa45f2a3fc70a01dfd3b66a` | TorchScript artifact documented in `README.md` | Artifact is fetched separately, SHA-256 verified, and adapted locally; it is not checked into Git. |

All revisions were resolved from the repositories' default branch heads during implementation on 2026-09-15.

