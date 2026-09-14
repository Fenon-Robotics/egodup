# Third-party notices

`src/egodup/vendor/temporal_network.py` is a narrow adaptation of `vcsl/vta.py` from VCSL, copyright (c) 2021 vcsl-owner, used under the MIT License. It removes unrelated DTW/DP/SPD/YOLO, multiprocessing, torch, numba and logging code, preserves TN coordinate semantics, replaces the removed IoU helper, and returns matched frame-index paths in addition to boxes.

`src/egodup/model.py` adapts the structure and model-equivalence procedure from `vsc/baseline/adapt_sscd_model.py`, copyright (c) Meta Platforms, Inc. and affiliates, used under the MIT License.

The full MIT license text appears in this repository's `LICENSE`; upstream revision and source details appear in `third_party/UPSTREAM.md`.

The SSCD checkpoint is downloaded on operator request and is not redistributed in this repository. Operators should independently review model terms for their use case.

