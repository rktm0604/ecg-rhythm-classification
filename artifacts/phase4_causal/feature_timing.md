# Phase 4 strictly causal feature timing

Prediction time is the reference annotated R-peak sample `R`.

- The waveform window is `signal[R-199:R+1]`: 200 samples indexed from R-199 through R, inclusive. It contains no samples after R.
- Only annotations whose symbols map through `pipeline.LABEL_MAP` are heartbeat positions. Unmapped/rhythm markers remain available in the annotation arrays but are never interval endpoints.
- `rr_prev` is the current valid heartbeat sample minus the previous valid heartbeat sample.
- `rr_local` is the mean of up to four intervals ending at or before the current heartbeat: the intervals before `rr_prev` plus `rr_prev` itself. It never includes a following interval.
- No `rr_next` feature is calculated.
- RF features are `rr_prev`, `rr_local`, `rr_prev / rr_local`, and 30 evenly spaced samples from the causal window. CNN receives the same causal 200-sample window.
- The first valid heartbeat in a record has no preceding heartbeat and is explicitly excluded. A mapped beat is also excluded when its 200-sample causal window crosses the record start, when the waveform contains non-finite samples, or when RR context is invalid. Exclusion counts are recorded by record and reason.

This is a secondary experiment with information available through the R-peak only. It is not identical to Phase 2, which uses a window extending 110 samples after R and includes future RR context.
