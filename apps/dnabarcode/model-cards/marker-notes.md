These notes are written by hand and included in the generated card.

### Design

- **Labels** are the 15 mitochondrial genes in MIDORI2 (CO1, CO2, CO3, Cytb,
  A6 and A8 (ATP synthase 6 and 8), ND1-ND6 and ND4L, and the large and
  small ribosomal RNAs).
- **Avoiding a shortcut.** The genes are sequenced from very different
  organisms: CO1 records are mostly insects, while Cytb and the ribosomal
  RNAs are mostly vertebrates. A model could learn "fish-like means Cytb"
  rather than what each gene looks like. To prevent this, each gene's
  training records are spread across as many orders as possible, and the
  results are reported per group of organisms.
- **Strand.** Sequences may be given as either strand. Every training
  record's other strand (reverse complement) is added as a *reverse* class,
  so the model recognises a sequence given backwards. Both strands of a
  query are predicted, and the strand whose best gene is more confident is
  kept; the evaluation measures how often this picks the right strand.
- **Training on fragments and calibration** follow the router (see its
  card): one random fragment of 75-600 bases per record, and a
  length-aware temperature fitted on validation sequences and fragments.

### Version history

- **v1-v2** (one training run, then calibration): 98.6% accurate overall,
  but only 62% for fungi, and the other strand was more confident for 2%
  of animal and 34% of fungal records. An analysis
  (`analysis/04_marker_errors.py`) showed why. Most fungal errors were the
  two ribosomal RNAs mistaken for each other, and fungal protein-coding
  genes mistaken for ND2 or ND5: records were spread across orders, and
  animals have far more orders, so fungi were scarce in training. When the
  other strand won, it was mostly predicted as a *different* gene (a model
  error), not the record's own gene (which would mean a record stored
  backwards).
- **v3-v4:** records took turns between groups of organisms before
  orders, and the reverse class was added. The reverse class worked: the
  other strand was now more confident for only 0.5% of animal and 1.8% of
  fungal records. Balancing groups did not: fungi stayed at about 60%
  accurate, and animals fell from 99.5% to 98.3%. Naive Bayes was dropped
  from the search (0.870 against 0.915 for logistic regression in v1).
- **v5 onwards:** records are spread across orders again (as in v1), with
  the reverse class kept.

### Fungi

Fungal sequences remain the weak point (about 60% accurate in v1-v4), and
more fungal training data did not help. A likely reason is biology rather
than sampling: fungal mitochondrial genes such as *cox1* (CO1), *cob*
(Cytb), *nad5* (ND5) and the large ribosomal RNA often contain many
introns, which can make up much of a record. A model that counts k-mers
across the whole record then sees mostly intron sequence, which these genes
share. This has not been tested here. For this site it matters little:
fungi are identified by ITS, not COI, and the marker classifier's main job
is to check animal COI.

### What it cannot do

It knows only these 15 genes. A sequence from another marker or genome is
still given one of them, so the site also checks how similar a sequence is
to the reference data, and says when a sequence is not recognised.
