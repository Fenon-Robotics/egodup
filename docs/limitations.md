# Limitations

- This is an experimental baseline with unvalidated `development-v1` thresholds.
- Only the first 240 sampled seconds of a pair are localized in 0.1.0; long-video window expansion is pending.
- Optional verification records uncalibrated status; it does not yet provide an independent descriptor family.
- Reverse playback, severe crops, picture-in-picture, synthetic edits and adversarial transforms are unsupported claims.
- Repetitive/static scenes can be ambiguous. Human review remains required.
- A missed retrieval candidate cannot be recovered by localization. Negative output is not proof of originality.
- The HTML report currently summarizes intervals and metrics; frame thumbnail materialization is planned after retention controls are finalized.

