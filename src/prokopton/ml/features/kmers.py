"""The old import path of ``KmerVectorizer``, which moved to
``prokopton.ml.bio.features.kmers`` in 0.2.0.

Models saved before then (the routers and marker classifiers in the MLflow
registry and in serving bundles) refer to this path, so it is kept until they
have been saved again under the new one.
"""

from prokopton.ml.bio.features.kmers import KmerVectorizer

__all__ = ["KmerVectorizer"]
