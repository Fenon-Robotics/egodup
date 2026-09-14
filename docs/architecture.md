# Architecture

The immutable identity boundary is the full-file SHA-256. A submission is a relationship to a media asset, so aliases remain visible while feature work is cached once. Sample selection uses decoded presentation timestamps and stores actual times rather than deriving them from nominal FPS.

The `vsc_raw_v1` descriptor profile keeps raw 512-dimensional float32 SSCD output. FAISS `IndexFlatIP` consumes those raw vectors. Temporal localization creates bounded normalized copies and calculates `Zq @ Zr.T`, preserving the upstream raw-retrieval/cosine-localization distinction.

The local database separates assets, submissions, feature sets, immutable index generations and runs. Feature arrays are published via staging followed by same-filesystem atomic rename. A file lock coordinates index writers.

