These notes are written by hand and included in the generated card.

### How the model was chosen

The search ran in rounds, each narrowed by the last; every candidate of
every round is kept in MLflow. Scores are the mean macro-F1 on validation.

1. **Naive Bayes and logistic regression, k = 5 to 8.** Naive Bayes did best
   at the smallest smoothing tried (0.01, score 0.786). The k-mer features
   are frequencies adding up to 1 per sequence, so useful smoothing is very
   small. Logistic regression did poorly (about 0.2): its features were too
   small in scale to learn from in 15 passes.
2. **Smaller smoothing, and logistic regression on features rescaled to
   unit length.** Naive Bayes reached 0.888 (k = 8); logistic regression
   reached 0.934 (k = 6), at the smallest regularisation tried.
3. **Logistic regression at k = 8 and less regularisation.** 0.949
   (k = 8, alpha 1e-8). Gains were shrinking (+0.015, then +0.006 for the
   last step in alpha), so the search stopped.
4. **Training on random fragments.** See below. Candidates were then chosen
   on validation fragments as well as whole sequences.

The ambiguity threshold for training records (0.01 to 0.10) made no real
difference in any round (about ±0.001).

### Short sequences: found on the test set

The first evaluated model (v3) scored 98.6% (known species) and 96.7%
(novel species, clean) on whole sequences, but was much less accurate on
short fragments, and overconfident: at 100 bases, 41% accurate with an
average confidence of about 62%. This was found with the test set, not the
validation set. Training on fragments fixes the method rather than tuning a
setting to the test scores, and from then on fragments were part of
validation, so model choice no longer depended on the test set; but the
test set did inform that change, which is why it is recorded here.

Training on fragments (one per record, 75-600 bases) raised accuracy at
every length, including whole sequences, but left the model underconfident
(average confidence 93.5% for 97.5% accuracy). Calibrating with a
length-aware temperature fixed that. The fitted length term was close to
zero: a single temperature (about 0.75) suits every length from 75 to 600
bases.

### Validation and test scores differ

Macro-F1 on novel species is lower on the test set than on validation. The
validation samples were balanced (capped per class) and left out records
with over 10% ambiguous bases; the test sets keep the natural mix, where
rare classes have few records and noisier scores. Choosing the best of many
candidates also makes the validation score slightly optimistic. The test
scores are the ones to rely on.

### Future work

- Logistic regression at k = 9 (each candidate takes 30-40 minutes here).
- A small convolutional network, as a stronger comparison.
- Fitting calibration on even shorter fragments, if inputs under 100 bases
  are ever to be accepted.
