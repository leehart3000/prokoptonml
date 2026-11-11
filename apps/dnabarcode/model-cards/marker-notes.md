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
- **Strand.** Sequences may be given as either strand. Rather than a
  separate model, both strands are predicted and the more confident one is
  kept; the evaluation measures how often this picks the right strand.
- **Training on fragments and calibration** follow the router (see its
  card): one random fragment of 75-600 bases per record, and a
  length-aware temperature fitted on validation sequences and fragments.

### What it cannot do

It knows only these 15 genes. A sequence from another marker or genome is
still given one of them, so the site also checks how similar a sequence is
to the reference data, and says when a sequence is not recognised.
