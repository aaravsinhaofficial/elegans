# Figure Guide

The runner generates the four requested figures plus a reversal diagnostic. PNG and
PDF versions contain identical panels.

1. **Cable and vigor trace.** True own-action gain, both pre-update Gaussian NLLs,
   display-smoothed unclipped evidence, operational influence, vigor, and action RMS.
   Lines are seed means; bands are pointwise 95% normal intervals across seeds. The
   rolling display window is 30 transitions for the 4,500-transition run.
2. **Matched-yoke distributions.** Seed-level phase means from the final 50% of each
   phase, accompanied by sensory-state variance. Points are seeds and boxes summarize
   their distribution. The dashed line is the analytic connected target.
3. **Probe recovery.** Paired capped recovery latency after probes are withdrawn at
   disconnection versus persistent 3% and 5% probes. All arms are exactly matched
   through the initial connected phase. Triangles mark right-censored runs at the 1,500-transition
   horizon; horizontal bars are medians.
4. **Gain/noise robustness.** Analytic information, mean unclipped prequential loss
   advantage, absolute calibration error, and mean within-seed connected-versus-yoked
   ROC-AUC. Each cell uses the held-out robustness seed block and the final 50% of
   each phase.
5. **Reversed mapping diagnostic.** Gain sign, action-aware squared error, influence,
   and learned action coefficient. It shows the transient mismatch separately from
   steady action influence.

Inferential intervals in `inference.csv` are 95% percentile intervals from 10,000
seed-level bootstrap resamples. Time points are not bootstrap units. Exact seed counts
and configuration hashes are recorded in `run_manifest.json`.
