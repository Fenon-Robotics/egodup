# Calibration

The active calibration identifier is `development-v1` and is explicitly provisional. Defaults require ten supported seconds, eight distinct query samples, 60% support coverage, median cosine 0.80, tenth-percentile cosine 0.65 and no unsupported internal gap above three seconds.

Before production use, split original recording sessions before deriving transformations; keep every derivative with its source. Include independent recordings of the same activity/workstation as hard negatives. Measure candidate recall separately from interval localization, and report sample counts and confidence intervals.

