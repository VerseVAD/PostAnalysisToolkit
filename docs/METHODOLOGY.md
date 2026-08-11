# Methodology

This document summarizes the toolkit's shared analytical rules. Tool-specific
choices are described in [TOOLS.md](TOOLS.md). The authoritative implementation
remains the readable Python source and the specification/metadata exported with
each run.

## Analytical unit and aggregation

For corpus analyses, the ordinary analytical unit is a work/poem. VerseVAD
first produces poem-level measurements. Corpus means are then **equal-work
means** of those poem-level values unless an output explicitly says otherwise.
A longer poem therefore does not receive more corpus weight merely because it
contains more matched tokens.

This does not prove that works are statistically independent. Researchers must
consider authorship, period, collection structure, textual transmission, and
sampling design.

## Lexical scopes and weighting

The corpus export can provide six compatible profiles:

1. All lexical tokens / Token-weighted
2. All lexical tokens / Type-weighted
3. Stopword-excluded / Token-weighted
4. Stopword-excluded / Type-weighted
5. Content words only / Token-weighted
6. Content words only / Type-weighted

Token weighting retains repetition: every eligible occurrence can contribute.
Type weighting gives each eligible distinct lexical type one contribution.
The scopes determine which tokens are eligible before weighting. A script never
silently substitutes a different scope or weighting.

## Coverage and missingness

Coverage records how much eligible evidence a resource actually matched. A
coverage threshold is applied to each requested metric before analysis. For a
pair, both measurements must pass. Missing values remain missing; they are not
converted to zero, and denominators are not silently borrowed from another
profile. Exports retain counts and coverage so exclusions can be audited.

## VAD resources and scales

Valence, arousal, and dominance may be available from multiple normative
resources. Each exact resource/profile combination is treated as its own
methodological variant. VerseVAD exports supported VAD means on a normalized
0–1 scale. The toolkit uses those exported values directly and does not
normalize them a second time. Other measures retain their exported scales.

## Correlation

Spearman rank correlation is primary because it estimates monotonic association
without requiring a linear relationship or normally distributed raw measures.
Optional Pearson correlation is a linear sensitivity check, not a replacement
for Spearman. Correlation does not imply causation.

The paired bootstrap resamples whole paired works, with replacement, using a
fixed recorded seed. Spearman ranks are recomputed inside every resample. The
percentile interval describes resampling uncertainty under this procedure; it
does not establish literary importance.

Benjamini–Hochberg false-discovery-rate adjustment is applied across the metric
pairs the researcher selected for that run. The selected pairs—not every
possible VerseVAD pairing—form the multiple-testing family.

The optional quadratic screen asks whether a curved relationship improves on a
linear term and whether an estimated turning point lies inside the observed
range. It is a guarded exploratory screen, not an automatic claim of a poetic
threshold, causal mechanism, or formal change point.

## Leave-one-out robustness

Leave-one-out (LOO) analysis removes each included work once and recomputes the
selected statistic. It reports the estimate range, maximum change, influential
work, and relevant jackknife diagnostics. LOO replicates are influence checks;
they are not separately bootstrapped and do not form a new FDR family.

## Methodological sensitivity

Sensitivity analysis compares the same construct across reasonable exported
resource, scope, and weighting variants. The recommended default is the
**common qualifying set**: every variant is calculated over the same works that
have usable evidence for all selected variants. This isolates methodological
movement from sample-composition movement.

The explicit available/pairwise alternative retains more observations but can
mix method effects with changes in which works qualify. Outputs report the
alignment mode and sample counts.

## Anomaly exploration

Anomaly tools describe corpus-relative unusualness. Single-metric extremes,
directional two-metric combinations, and broad multi-metric profiles answer
different questions. Broad scores exclude or balance technical, cumulative,
and unusually verbose evidence so one module or poem length does not dominate.
An unusual work is not necessarily erroneous, pathological, or more valuable.

## Corpus comparison

Corpus comparison uses the same resource, metric, scope, weighting, and scale
in every corpus. Corpus descriptions use work-level distributions. Pairwise
results report signed and absolute raw differences **within a metric** plus
bootstrap intervals for mean differences.

Raw differences should not be ranked across unrelated scales. Cliff's delta is
the scale-independent effect size used for cross-metric ranking. It estimates
how often a randomly selected work from one corpus exceeds a randomly selected
work from another, minus the reverse probability; ties contribute zero. The
comparison tool intentionally does not add inferential p-values.

## Single-poem analysis

`single.py` treats the Complete Audit's exported whole-poem profile values as
authoritative. It derives line, stanza, rolling-window, contributor, and
influence views only when its observation adapter reproduces the exported mean
within numerical tolerance. It preserves original line/stanza evidence and
does not label rolling movement as a statistically detected volta or change
point.

## Types of claim

- **Description:** what a measured distribution looks like.
- **Correlation:** how two measurements co-vary.
- **Sensitivity:** how an answer changes under reasonable methodological choices.
- **Robustness:** how dependent an answer is on individual works.
- **Anomaly exploration:** which works are unusual relative to a selected corpus.
- **Effect-size comparison:** how strongly two corpus distributions separate.
- **Inference/interpretation:** a researcher-led argument requiring assumptions,
  context, theory, and close reading beyond the software output.
