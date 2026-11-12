The specialist was developed in six rounds, followed by curation of the reference data, each run on the same validation
data. Trials 1 and 2 are recorded in `analysis/README.md`
(`05_specialist_trial.py`); the evaluations used `prokopton-ml evaluate
--specialist culicidae`.

### Trial 1: starting thresholds

Other flies were all kept out, and 95.0% of known species were right, but
held-out species were often given the name of a sister species. The trial
found three faults:

- coverage was measured on the query only, so whole CO1 genes failed
  against barcode-length references;
- many copies of one species could fill every candidate slot;
- species that COI does not separate (such as *Culex pipiens* and
  *Cx. quinquefasciatus*) were named as one or the other.

### Trial 2: matching fixes and new wording

Coverage now uses whichever sequence lies more fully inside the alignment;
identical sequences are kept once per species, with at most 3 candidates per
species; species within a margin are reported together; and a barcode-gap
check reports a *closest match* when other species are as close. Known
species reached 99.3% right, with 0.3% wrongly named (counted per record).
Every answer was reworded to describe our reference data, not the
organism's scientific status.

### Evaluation 1: tuning per species

Scoring each species equally showed known species wrongly named 2.5–4.3% of
the time, whatever the threshold, and held-out species without a near-copy
in training were almost always right. The ranking rule had a flaw: the
membership threshold did not change the objective, so the first (88%) was
chosen, letting in about 1% of other flies.

### Evaluation 2: ranking fixed, wrong answers listed

Keeping out other flies now comes first. The listed wrong answers showed
five causes: sister species COI does not separate (most), inconsistent
genus names (*Aedes* and *Ochlerotatus*), mislabelled or contaminated
records, a spelling variant, and one record whose region barely overlapped
the references.

### Evaluation 3: consistent names

Genera split from *Aedes* are reported as *Aedes* in the broad sense, a
spelling variant is mapped, the *Anopheles maculipennis* complex and
*An. funestus* group were added, and placeholder references without a genus
were left out. Other flies let in fell to 0.1%, and held-out species without
a near-copy rose to 94.4% right, but known species were still wrongly named
3.2% of the time (by species): the remaining errors are sister species and
mislabelled records that names cannot fix.

### Evaluation 4: choosing caution

The known-species limit was raised from 1%, which no threshold reached.
At 3.5% the rule chose a species threshold of 99.0%; at 3% it chooses
99.5%. We chose 3%: a confident wrong name costs a user more than a less
specific answer. At 99.5%, about 6 points fewer known species are named
exactly (on validation, averaged over species: 51.0% against 56.9%), but
held-out species are wrongly named less often (17.8% against 23.2%), and
known species too (2.5% against 3.2%).

### Curation: leaving out mislabelled records

Raising the serving reference to 200 records per species drew in records
such as KP896564.1, labelled *Aedes albopictus* but 99.6% identical to
*Ae. aegypti*. Two automatic rules were tried. The first (a majority vote of
the closest records) missed groups of mislabels that vouch for each other,
and removed sister species. The second (unlike the species as a whole by
median k-mer similarity) removed 690 records, mostly real sister and cryptic
species, probably because k-mer similarity also falls when records cover
different stretches of COI. The rule kept judges only near-identity to
**another genus**, which is almost always an error. Within a genus,
records are checked against an authoritative source instead: the COI genes
of NCBI's curated RefSeq mitogenomes (one or more for about 170 mosquito
species). A record at least 99% identical (by alignment) to another
species' RefSeq COI, but under 95% to every RefSeq COI of its own species,
is left out, unless the two species are in a configured complex. Remaining
errors found later are listed by hand in the config, with reasons.

Possible next steps:
comparing the best mosquito match with the best other-fly match, rather
than a fixed membership threshold; checking each record against its own
species by alignment rather than k-mers, to find errors within a genus
automatically; and adding BOLD data and BINs.
