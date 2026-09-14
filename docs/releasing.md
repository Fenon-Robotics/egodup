# Release checklist

1. Build and install wheel in a clean Python 3.11 environment.
2. Run unit/integration tests and raw FAISS parity.
3. Fetch and adapt the pinned checkpoint with checksum and real-image parity validation.
4. Exercise exact compare, transformed compare, index, scan, rebuild, resume and offline HTML output.
5. Run CPU portability and separately recorded single-CUDA-device validation.
6. Record actual hardware/software measurements without extrapolated throughput claims.
7. Confirm the VM used for release validation has been terminated.

