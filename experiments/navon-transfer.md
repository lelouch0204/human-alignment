# Navon transfer across rendering conditions

This exploratory follow-up uses the same pinned public embeddings as
`navon-followup.md`. The earlier decoding results are already known.

## Design fixed before running

- Primary representation: `pre_logits`; report `triplet_logits` separately.
- For each of s10, s14, s18, train on the other two conditions and test on the
  held-out condition. Size and spacing vary jointly, so this tests rendering
  transfer, not an isolated effect of scale.
- Use the same five seed-0 nuisance-letter folds as the prior experiment. Hold
  out local identities for global decoding, and global identities for local
  decoding. Exclude congruent compounds. Both positions are included.
- Choose ridge alpha from powers of ten 1e-6 through 1e4 using only the outer
  training data. Within that data, cross three inner nuisance-group folds with
  both directions of transfer between the two source conditions. Each of the
  six inner splits holds out both a nuisance group and a rendering condition.
  Pool inner validation accuracy; prefer larger alpha on ties.
- Refit on both source conditions and all outer training nuisance groups.
- Evaluate that same fitted classifier on (a) unseen nuisance letters in the
  held-out rendering condition and (b) those identical letter combinations and
  positions in both source conditions. These in-domain controls use no extra
  fitted models and exclude all outer training nuisance identities.
- Compare out-of-domain accuracy with the mean in-domain control accuracy on
  matched global/local identities and positions. Also report each held-out
  condition separately. This controls model fitting and letter composition;
  differences still include intrinsic rendering difficulty.
- Save held-out predictions, per-fold alpha and validation scores, domain-level
  scores, matched domain gaps, checkpoint contrasts, hashes, and a static figure.
  Treat results descriptively: letters, images, and folds are dependent.

The same image appears once as an out-of-domain test image and twice as an
in-domain control across the three held-out-condition fits. These appearances
must not be counted as independent observations. No human-agreement claim is
possible from these embeddings alone. Source preprocessing limitations remain.

## Run

```bash
uv run --locked --only-group analysis python src/human_alignment/navon_transfer.py
uv run --locked --only-group analysis python -m unittest discover -s tests -p 'test_navon_transfer.py'
```

Inputs are reused from `outputs/navon_followup/data`. Results are saved under
ignored `outputs/navon_transfer`. The default runs the primary representation;
`--representations pre_logits triplet_logits` runs both separately. No downloads,
heavy installs, or model inference are required.
