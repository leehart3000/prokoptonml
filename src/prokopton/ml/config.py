"""Configuration models for ProkoptonML applications.

Each application (for example dnabarcode) is described by one TOML file that is
validated against ``AppConfig``. Relative paths in the file are resolved from
the current working directory, normally the repository root.
"""

from pathlib import Path
from typing import Literal

from prokopton.core.config import ConfigModel
from pydantic import Field, field_validator, model_validator

__all__ = [
    "ALLOWED_URL_SCHEMES",
    "AppConfig",
    "ClassifierCalibrationConfig",
    "ClassifierCardConfig",
    "ClassifierConfig",
    "ClassifierEvaluationConfig",
    "ClassifierSampleConfig",
    "ClassifierServingConfig",
    "ClassifierTrainingConfig",
    "ComplexConfig",
    "DataConfig",
    "DataPathsConfig",
    "LeakageConfig",
    "Midori2SourceConfig",
    "MlflowConfig",
    "SpecialistConfig",
    "SpecialistEvaluationConfig",
    "SplitConfig",
    "UnknownClassifierError",
    "ValidationConfig",
]

ALLOWED_URL_SCHEMES: tuple[str, ...] = ("https://", "file://")
"""URL schemes accepted for downloads: HTTPS, plus local files for testing."""


def _check_url(value: str) -> str:
    if not value.startswith(ALLOWED_URL_SCHEMES):
        msg = f"URL must start with one of {ALLOWED_URL_SCHEMES}: {value!r}"
        raise ValueError(msg)
    return value


class Midori2SourceConfig(ConfigModel):
    """Where to get a MIDORI2 release, and which genes to ingest from it."""

    type: Literal["midori2"] = "midori2"
    release: str = Field(pattern=r"^GB\d+$")
    doi: str
    licence: str
    archive_url: str
    checksums_url: str
    archive_name: str
    member_template: str
    genes: tuple[str, ...] = Field(min_length=1)

    @field_validator("archive_url", "checksums_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        return _check_url(value)

    @field_validator("member_template")
    @classmethod
    def _validate_member_template(cls, value: str) -> str:
        if "{gene}" not in value:
            msg = "member_template must contain '{gene}'"
            raise ValueError(msg)
        return value


class DataPathsConfig(ConfigModel):
    """Local folders for downloaded (raw), converted (interim) and validated
    (processed) data."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    splits_dir: Path
    leakage_dir: Path
    samples_dir: Path
    models_dir: Path


class ValidationConfig(ConfigModel):
    """Rules for the validation stage.

    ``max_ambiguous_fraction`` defaults to 1.0 (no filtering): the stage only
    reports how many records exceed each of ``ambiguity_report_thresholds``.
    The ambiguity threshold is treated as a hyperparameter and applied by the
    training stage to training data only, so evaluation still uses realistic,
    imperfect sequences.
    """

    phylum_reference: Path
    min_length: int = Field(default=100, ge=1)
    max_length: int = Field(default=2000, ge=1)
    max_ambiguous_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguity_report_thresholds: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
    short_length: int = Field(default=200, ge=1)
    top_shared_examples: int = Field(default=10, ge=0)


class DataConfig(ConfigModel):
    """Data settings for an application."""

    source: Midori2SourceConfig
    paths: DataPathsConfig


class SplitConfig(ConfigModel):
    """How records are assigned to training, validation and test sets.

    Assignment uses a hash of each record's key (the species name for novel
    holdouts, the sequence otherwise) mixed with ``salt``, so it is reproducible
    and stable when data is added. Changing ``salt`` reshuffles every split.
    """

    salt: str = Field(default="prokopton-v1", min_length=1)
    novel_val_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    novel_test_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    val_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    test_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    novel_species_genes: tuple[str, ...] = ("CO1",)
    report_families: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_fractions(self) -> "SplitConfig":
        if self.novel_val_fraction + self.novel_test_fraction > 1.0:
            msg = "novel_val_fraction + novel_test_fraction must not exceed 1"
            raise ValueError(msg)
        if self.val_fraction + self.test_fraction >= 1.0:
            msg = "val_fraction + test_fraction must be below 1 (train needs data)"
            raise ValueError(msg)
        return self


DEFAULT_VSEARCH_COMMAND: tuple[str, ...] = (
    "pixi",
    "exec",
    "--channel",
    "conda-forge",
    "--channel",
    "bioconda",
    "--spec",
    "vsearch>=2.28,<3",
    "vsearch",
)
"""Runs VSEARCH from bioconda in a temporary environment, without changing the
project's own environment."""


class LeakageConfig(ConfigModel):
    """How the leakage stage finds near-duplicates of training sequences.

    Records in ``query_splits`` with a match of at least ``min_identity`` in
    ``reference_split`` are listed, so evaluation can exclude them from
    novel-species scores. The default VSEARCH settings stop at the first
    match and give up after a few non-matching candidates, which found about
    95% of the leaks found by a thorough search in about a third of the time
    (see ``analysis/README.md``).
    """

    genes: tuple[str, ...] = ("CO1",)
    query_splits: tuple[str, ...] = ("val_novel", "test_novel")
    reference_split: str = "train"
    min_identity: float = Field(default=0.99, gt=0.0, le=1.0)
    maxaccepts: int = Field(default=1, ge=1)
    maxrejects: int = Field(default=8, ge=1)
    threads: int = Field(default=8, ge=1)
    chunk_size: int = Field(default=250_000, ge=1)
    vsearch_command: tuple[str, ...] = Field(
        default=DEFAULT_VSEARCH_COMMAND, min_length=1
    )


class ClassifierSampleConfig(ConfigModel):
    """How a classifier's training and validation samples are drawn.

    Each class (for the router, an animal order or a group for non-animals;
    for the marker classifier, a gene) is capped, so a few huge classes do
    not dominate. Within a class, records are taken from as many different
    species (or orders, with ``diversity = "order"``) as possible.
    ``max_ambiguous_fraction`` should be the loosest value that training will
    try; stricter thresholds are applied later, when training.
    """

    salt: str = Field(default="prokopton-sample-v1", min_length=1)
    # Spread each class's records across as many species, or as many orders,
    # as possible; "group_order" first takes turns between groups of
    # organisms (Metazoa, Fungi, ...) and then between orders within each,
    # so no group dominates a gene-wide class
    diversity: Literal["species", "order", "group_order"] = "species"
    train_cap: int = Field(default=2000, ge=1)
    eval_cap: int = Field(default=200, ge=1)
    min_class_records: int = Field(default=20, ge=1)
    max_ambiguous_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    eval_splits: tuple[str, ...] = ("val_known", "val_novel")
    exclude_near_duplicates: bool = True


class ClassifierTrainingConfig(ConfigModel):
    """The models and settings a classifier's training stage compares.

    Every combination of ``k_values``, ``ambiguity_thresholds`` and
    ``naive_bayes_alphas`` is tried with naive Bayes (fast). Logistic
    regression (slower) is tried for each of ``logistic_k_values`` and
    ``logistic_alphas``, at ``logistic_ambiguity_threshold``, on features
    rescaled to unit length.

    With ``fragments_per_record`` above 0, training also gets that many
    random fragments of each record, of lengths from ``fragment_min_length``
    to ``fragment_max_length``, so the model learns short inputs too. Each
    validation sample then gets one such fragment per record as well.

    With ``reverse_label`` set, the reverse complement (the other strand)
    of every training record is added too, labelled ``reverse_label``, so
    the model learns to recognise a sequence given as the other strand.

    The best candidate is chosen by ``selection_score``: the mean of macro-F1
    on the known-species and novel-species validation samples (and their
    fragments, if any). Leave ``k_values`` or ``naive_bayes_alphas`` empty to
    try logistic regression only.
    """

    k_values: tuple[int, ...] = (5, 6, 7, 8)
    ambiguity_thresholds: tuple[float, ...] = Field(default=(0.03, 0.10), min_length=1)
    naive_bayes_alphas: tuple[float, ...] = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    logistic_k_values: tuple[int, ...] = (6,)
    logistic_ambiguity_threshold: float = Field(default=0.03, ge=0.0, le=1.0)
    logistic_alphas: tuple[float, ...] = (1e-7, 1e-6)
    logistic_max_iter: int = Field(default=15, ge=1)
    fragments_per_record: int = Field(default=0, ge=0)
    fragment_min_length: int = Field(default=75, ge=1)
    fragment_max_length: int = Field(default=600, ge=1)
    reverse_label: str | None = None
    threads: int = Field(default=8, ge=1)
    random_state: int = 273
    experiment: str = Field(default="router", min_length=1)
    registered_model: str = Field(default="router", min_length=1)
    alias: str = Field(default="candidate", min_length=1)

    @model_validator(mode="after")
    def _check_fragment_lengths(self) -> "ClassifierTrainingConfig":
        naive_bayes = self.k_values and self.naive_bayes_alphas
        logistic = self.logistic_k_values and self.logistic_alphas
        if not naive_bayes and not logistic:
            msg = "The search has no candidates: give naive Bayes or logistic settings"
            raise ValueError(msg)
        if self.fragment_min_length > self.fragment_max_length:
            msg = (
                f"fragment_min_length ({self.fragment_min_length}) is above "
                f"fragment_max_length ({self.fragment_max_length})"
            )
            raise ValueError(msg)
        return self


class ClassifierCalibrationConfig(ConfigModel):
    """How a trained classifier's confidence is calibrated.

    A length-aware temperature (see ``prokopton.ml.training.calibration``)
    is fitted on the validation samples: their whole sequences, plus one
    random fragment of ``fragment_min_length`` to ``fragment_max_length``
    bases per record. The calibrated model is registered as a new version
    and given the training alias.
    """

    fragment_min_length: int = Field(default=75, ge=1)
    fragment_max_length: int = Field(default=600, ge=1)
    threads: int = Field(default=8, ge=1)
    # Differs from training's, so these fragments are new ones
    random_state: int = 274

    @model_validator(mode="after")
    def _check_fragment_lengths(self) -> "ClassifierCalibrationConfig":
        if self.fragment_min_length > self.fragment_max_length:
            msg = (
                f"fragment_min_length ({self.fragment_min_length}) is above "
                f"fragment_max_length ({self.fragment_max_length})"
            )
            raise ValueError(msg)
        return self


class ClassifierEvaluationConfig(ConfigModel):
    """How the model chosen by training is measured on the test sets.

    All of each test split is predicted. Records whose label the model never
    saw (classes dropped by the sample stage) are counted, not scored. For
    novel-species records, scores are also given separately for records with
    and without a near-copy in training (from the leakage stage).

    For the "how short can you go?" chart, a ``fragment_fraction`` share of
    each split (chosen by a salted hash) is cut to each of
    ``fragment_lengths`` at a random position, and predicted again.
    """

    splits: tuple[str, ...] = Field(default=("test_known", "test_novel"), min_length=1)
    novel_splits: tuple[str, ...] = ("test_novel",)
    fragment_lengths: tuple[int, ...] = (50, 75, 100, 150, 200, 300, 400, 500)
    fragment_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    ambiguity_edges: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.10)
    confidence_thresholds: tuple[float, ...] = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
    calibration_bins: int = Field(default=15, ge=1)
    batch_size: int = Field(default=20_000, ge=1)
    threads: int = Field(default=8, ge=1)
    # Also check, on the fragment records, that the right strand is the
    # more confident one (forward against reverse complement)
    orientation: bool = False
    salt: str = Field(default="router-evaluation", min_length=1)
    random_state: int = 273


class ClassifierServingConfig(ConfigModel):
    """How a classifier's predictions are presented to users.

    Sequences shorter than ``min_length`` bases are declined; up to
    ``short_length`` they are routed with a note that accuracy is lower.
    Predictions at or above ``confident_threshold`` are shown as confident.
    These come from the evaluation's "how short can you go?" and coverage
    tables, and are repeated in the model card.
    """

    min_length: int = Field(default=100, ge=1)
    short_length: int = Field(default=300, ge=1)
    confident_threshold: float = Field(default=0.9, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_lengths(self) -> "ClassifierServingConfig":
        if self.min_length > self.short_length:
            msg = (
                f"min_length ({self.min_length}) is above "
                f"short_length ({self.short_length})"
            )
            raise ValueError(msg)
        return self


class ClassifierCardConfig(ConfigModel):
    """The model card's place, and the parts of it written by hand.

    ``title``, ``task`` and ``intended_use`` describe the model;
    ``training_note`` is added to the training section (for example, to say
    that earlier search rounds are in the notes); ``limitations`` are added
    to those the card works out from the results.
    ``notes`` (optional) is a Markdown file included as the card's
    "Development notes" section.
    """

    output_dir: Path = Path("model-cards")
    notes: Path | None = None
    title: str = "Model card"
    task: str = ""
    input: str = "A DNA sequence (A, C, G, T; ambiguous bases allowed)"
    intended_use: str = ""
    training_note: str = ""
    limitations: tuple[str, ...] = ()


class ClassifierConfig(ConfigModel):
    """Settings for one classifier, such as the router or the marker classifier.

    ``label`` says what is predicted: ``group_order`` (the order for animals,
    the group otherwise) or ``gene`` (which of ``genes`` a sequence comes
    from). Records are drawn from the splits of each of ``genes``.
    """

    label: Literal["group_order", "gene"] = "group_order"
    genes: tuple[str, ...] = Field(default=("CO1",), min_length=1)
    sample: ClassifierSampleConfig = Field(default_factory=ClassifierSampleConfig)
    training: ClassifierTrainingConfig = Field(default_factory=ClassifierTrainingConfig)
    calibration: ClassifierCalibrationConfig = Field(
        default_factory=ClassifierCalibrationConfig
    )
    evaluation: ClassifierEvaluationConfig = Field(
        default_factory=ClassifierEvaluationConfig
    )
    serving: ClassifierServingConfig = Field(default_factory=ClassifierServingConfig)
    card: ClassifierCardConfig = Field(default_factory=ClassifierCardConfig)


class MlflowConfig(ConfigModel):
    """Where MLflow keeps experiment runs and model files.

    The default keeps everything in the git-ignored ``data/`` folder: run
    metadata and the model registry in a SQLite database, and model files in
    an artifact folder.
    """

    tracking_uri: str = "sqlite:///data/mlflow/mlflow.db"
    artifact_dir: Path = Path("data/mlflow/artifacts")


class ComplexConfig(ConfigModel):
    """A group of species that one marker cannot tell apart.

    When a specialist's best matches include members of a complex, it names
    the complex, and says how members can be told apart (``resolve``), with
    a source for that advice.
    """

    name: str = Field(min_length=1)
    members: tuple[str, ...] = Field(min_length=2)
    resolve: str = Field(min_length=1)
    reference: str = ""


class SpecialistEvaluationConfig(ConfigModel):
    """How a specialist's thresholds are tuned and its accuracy measured.

    Up to ``per_species`` records of each species are drawn from each set
    (at most ``max_records`` per set), so species with many records do not
    dominate. Each record is aligned once; every combination of the
    ``*_grid`` values is then scored on the validation sets.

    Combinations are ranked by, in order: letting in background records
    at most ``max_background_admitted`` of the time; wrongly naming known
    species at most ``max_known_wrong`` of the time (averaged over
    species); the objective, *known species named exactly* minus
    *held-out species wrongly named* (both averaged over species); and,
    to break ties, the most held-out species right. The best is measured
    once on the test sets. If it misses a limit, the report says which.
    """

    per_species: int = Field(default=3, ge=1)
    max_records: int = Field(default=1500, ge=1)
    salt: str = Field(default="prokopton-specialist-evaluation-v1", min_length=1)
    workers: int = Field(default=8, ge=1)
    species_identity_grid: tuple[float, ...] = Field(
        default=(0.975, 0.98, 0.985, 0.99, 0.995), min_length=1
    )
    group_margin_grid: tuple[float, ...] = Field(
        default=(0.005, 0.01, 0.015), min_length=1
    )
    member_identity_grid: tuple[float, ...] = Field(
        default=(0.88, 0.90, 0.92), min_length=1
    )
    max_known_wrong: float = Field(default=0.01, ge=0.0, le=1.0)
    max_background_admitted: float = Field(default=0.005, ge=0.0, le=1.0)


class SpecialistConfig(ConfigModel):
    """A specialist: species-level identification within one group.

    The reference set is drawn from the training split of ``gene``: records
    whose ``target_rank`` is ``target`` (up to ``per_species`` per species),
    and a background of other records whose ``background_rank`` is
    ``background``, so the specialist can tell when a sequence is not in its
    group. Close relatives (``close_relatives``, by family) are kept with up
    to ``per_species`` records per species; other background families are
    capped at ``background_per_family``.

    Identical sequences of the same species are kept once. A query is
    compared with the reference by k-mer similarity; the ``candidates`` most
    similar are aligned (at most ``candidates_per_species`` from any one
    species, so many copies of one species cannot crowd out the others),
    and the outcome follows from their percent identity (see
    ``prokopton.ml.specialists.identify``). A match counts only if it covers
    ``min_coverage`` of the query or the reference and aligns at least
    ``min_aligned`` bases. Species within ``group_margin`` of the best match
    are reported together.

    For ``neighbour_references`` records of each named species, the
    reference stage also finds the nearest *other* named species (its
    barcode gap). A query no closer to a species than that species'
    nearest neighbour is reported as a closest match, not a species.
    """

    gene: str = "CO1"
    target_rank: Literal["order", "family", "genus"] = "family"
    target: str = Field(min_length=1)
    background_rank: Literal["order", "family"] = "order"
    background: str = Field(min_length=1)
    close_relatives: tuple[str, ...] = ()
    per_species: int = Field(default=20, ge=1)
    background_per_family: int = Field(default=100, ge=1)
    max_ambiguous_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    salt: str = Field(default="prokopton-specialist-v1", min_length=1)
    k: int = Field(default=8, ge=1)
    candidates: int = Field(default=30, ge=1)
    candidates_per_species: int = Field(default=3, ge=1)
    min_coverage: float = Field(default=0.8, gt=0.0, le=1.0)
    min_aligned: int = Field(default=100, ge=1)
    neighbour_references: int = Field(default=3, ge=1)
    member_identity: float = Field(default=0.90, gt=0.0, le=1.0)
    genus_identity: float = Field(default=0.95, gt=0.0, le=1.0)
    species_identity: float = Field(default=0.98, gt=0.0, le=1.0)
    group_margin: float = Field(default=0.01, ge=0.0, le=0.1)
    complexes: tuple[ComplexConfig, ...] = ()
    evaluation: SpecialistEvaluationConfig = Field(
        default_factory=SpecialistEvaluationConfig
    )

    @model_validator(mode="after")
    def _check_identities(self) -> "SpecialistConfig":
        if not self.member_identity <= self.genus_identity <= self.species_identity:
            msg = "Need member_identity <= genus_identity <= species_identity"
            raise ValueError(msg)
        return self


class UnknownClassifierError(ValueError):
    """Raised when an app has no classifier of the name asked for."""


class AppConfig(ConfigModel):
    """Top-level configuration for a ProkoptonML application."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    data: DataConfig
    validation: ValidationConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    router: ClassifierConfig = Field(default_factory=ClassifierConfig)
    marker: ClassifierConfig | None = None
    specialists: dict[str, SpecialistConfig] = Field(default_factory=dict)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)

    def classifier(self, name: str) -> ClassifierConfig:
        """Return a classifier's settings by name (``router`` or ``marker``).

        Raises:
            UnknownClassifierError: If the app has no classifier of that name.
        """
        found = {"router": self.router, "marker": self.marker}.get(name)
        if found is None:
            msg = f"This app has no {name!r} classifier configured"
            raise UnknownClassifierError(msg)
        return found

    def specialist(self, name: str) -> SpecialistConfig:
        """Return a specialist's settings by name.

        Raises:
            UnknownClassifierError: If the app has no specialist of that name.
        """
        if name not in self.specialists:
            msg = f"This app has no {name!r} specialist configured"
            raise UnknownClassifierError(msg)
        return self.specialists[name]
