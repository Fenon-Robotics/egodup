# Measured validation

Validation was run on 2026-09-15 using a Lambda Cloud `gpu_1x_a10` instance. The instance downloaded the complete locked dependency environment and the SSCD checkpoint itself, verified checkpoint SHA-256 `9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56`, ran the experiment, and was then confirmed `terminated` through the Lambda API.

| Environment | Workload | Result |
|---|---|---|
| NVIDIA A10 23,028 MiB, driver 570.148.08 | 12 automated tests | 12 passed in 3.55 s |
| NVIDIA A10, CUDA, 18 s synthetic transformed pair | End-to-end compare | 3.20 s total; 1.85 s inference |
| Same pair | Localized evidence | 19 samples; 97.4% coverage; median cosine 0.902; p10 cosine 0.885 |
| Same environment | Index → scan → rebuild → report → wheel | Completed |
| CPU portability host, same pair | End-to-end compare | 240.09 s total; 214.65 s inference |

The fixture changes resolution (640×360 to 480×270), frame rate (24 to 30), color, and H.264 compression. It is a deterministic engineering fixture, not evidence of accuracy on egocentric industrial footage. No throughput or false-alert claims are inferred from this single pair.

