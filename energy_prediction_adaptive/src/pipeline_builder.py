"""
Pipeline Builder and Bundle Serialization Module for SURYOKTI Adaptive Architecture.
Assembles the fitted AdaptivePreprocessor, winning estimator, and schema metadata
into a single deployable inference bundle.
"""
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils import setup_logger, save_artifact, save_json
from src.adaptive_preprocessor import AdaptivePreprocessor

logger = setup_logger("pipeline_builder")


class AdaptiveInferenceBundle:
    """Encapsulates preprocessor, winning model, and schema metadata for inference."""

    def __init__(
        self,
        preprocessor: AdaptivePreprocessor,
        estimator: Any,
        metadata: Dict[str, Any]
    ):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.metadata = metadata

    def predict(self, df):
        """Transform input dataframe and predict."""
        X_trans = self.preprocessor.transform(df)
        return self.estimator.predict(X_trans)


def build_and_save_bundle(
    preprocessor: AdaptivePreprocessor,
    estimator: Any,
    metadata: Dict[str, Any],
    save_dir: Path
) -> Path:
    """Build and serialize the inference bundle and metadata JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    bundle = AdaptiveInferenceBundle(
        preprocessor=preprocessor,
        estimator=estimator,
        metadata=metadata
    )

    bundle_path = save_dir / "adaptive_model_bundle.joblib"
    meta_path = save_dir / "adaptive_model_metadata.json"

    save_artifact(bundle, bundle_path)
    save_json(metadata, meta_path)

    logger.info(f"Serialized deployable AdaptiveInferenceBundle to {bundle_path}")
    return bundle_path

