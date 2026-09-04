# Repeat-purchase label construction

This is the next development step after transaction cleaning. It creates a
supervised target for a customer repeat-purchase model and does not create
features, train models, or call Airflow, Feast, MLflow, or Kubernetes.

The builder reads `purchases.csv`, never `events.csv`, for the target. The
event view remains available for later event-derived features such as return
and cancellation rates.

## Label definition

For each explicit cutoff timestamp:

1. The observation window is `[cutoff - 90 days, cutoff)`.
2. The prediction window is `[cutoff, cutoff + 30 days)`.
3. A customer is eligible when they have at least one completed purchase in
   the observation window.
4. Label `1` means the eligible customer has at least one completed purchase
   in the prediction window; otherwise the label is `0`.

The half-open boundaries make the cutoff deterministic and prevent a purchase
at the cutoff from leaking into the observation history. The builder requires
the full observation and prediction windows to be covered by the input data;
it fails rather than producing censored labels.

## Run it

```powershell
python projects/retail-repeat-purchase/src/build_labels.py --help
python projects/retail-repeat-purchase/src/build_labels.py `
  --purchases projects/retail-repeat-purchase/data/processed/purchases.csv `
  --output projects/retail-repeat-purchase/data/processed/labels.csv `
  --summary projects/retail-repeat-purchase/data/processed/labels-summary.json `
  --cutoff 2011-11-09T00:00:00
```

Use multiple `--cutoff` arguments for temporal snapshots. Keep cutoffs
explicit and documented. The output contains only `customer_id`, the window
timestamps, and `label`; future purchase counts are not written as columns
that could accidentally become model features. The JSON summary records input
and output hashes, windows, eligible-customer counts, and class counts.

Run tests with:

```powershell
python -m unittest discover projects/retail-repeat-purchase/tests
```

## Recorded local run

Using the cleaned purchase input with SHA-256
`f2590d52459f71a90a74070b4e18437a2ff5605665f938830f71d0fdbdf14b28` and the
cutoff `2011-11-09T00:00:00` produced 2,566 eligible customer snapshots:
1,103 positives and 1,463 negatives. The generated label CSV had SHA-256
`1546fcfc1cb0c6ce040626e305ad8e16bc10c31bb75529c03c0d0a319d037155`; the
summary JSON is ignored with processed data and recreates these counts and
digests on each run.
