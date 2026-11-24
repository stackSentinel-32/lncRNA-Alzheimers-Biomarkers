import numpy as np
import pandas as pd

from scripts.ml.data_builder import build_dataset
from scripts.ml.feature_selection import make_lasso_selector
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


def test_build_dataset_excludes_non_binary_diagnoses(tmp_path):
    """Only AD and Control samples should enter the binary ML dataset."""
    root = tmp_path / "data"

    matrix_dir = root / "processed" / "stage10"
    metadata_dir = root / "interim" / "GSE63061"
    annotation_dir = root / "annotations"

    matrix_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    sample_ids = ["S1", "S2", "S3", "S4"]
    X = pd.DataFrame(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]],
        index=sample_ids,
        columns=["P1", "P2"],
    )
    X.to_parquet(
        matrix_dir / "GSE63061_validated_lncrna.parquet"
    )

    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "dataset_id": ["GSE63061"] * 4,
        "diagnosis": ["AD", "Control", "MCI", "Unknown"],
        "age": [70, 65, 68, 72],
        "sex": ["F", "M", "F", "M"],
        "batch": ["A"] * 4,
        "tissue": ["blood"] * 4,
    })
    metadata.to_csv(metadata_dir / "metadata.csv", index=False)

    mapping = pd.DataFrame({
        "probe_id": ["P1", "P2"],
        "mapping_status": ["VALIDATED_UNIQUE_LNCRNA"] * 2,
        "validated_gene": ["GENE1", "GENE2"],
        "exact_genes": ["GENE1", "GENE2"],
        "within_two_substitutions": ["GENE1", "GENE2"],
        "sense_exact": [True, True],
        "conflicting_sequence_or_annotation": [False, False],
        "sequence": ["A" * 50, "C" * 50],
        "platform": ["GPL10558"] * 2,
    })
    mapping.to_csv(
        annotation_dir / "GSE63061_probe_mapping.csv",
        index=False,
    )

    result = build_dataset(root, "GSE63061")

    assert result["X_lncrna"].shape == (2, 2)
    assert result["X_combined"].shape == (2, 4)
    assert result["y"].to_dict() == {"S1": 1, "S2": 0}
    assert result["feature_manifest"].shape[0] == 2


def test_lasso_pipeline_is_cloneable():
    """The unfitted selector must support sklearn cloning."""
    pipeline = make_lasso_selector("lncrna", C=0.1)
    cloned = clone(pipeline)

    assert not hasattr(cloned.named_steps["selection"], "estimator_")


def test_lasso_pipeline_fits_training_data():
    """The selector should fit and select features on training data."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.normal(size=(60, 8)),
        columns=[f"P{i}" for i in range(8)],
    )
    y = pd.Series([0, 1] * 30)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for train_idx, _ in cv.split(X, y):
        pipeline = clone(make_lasso_selector("lncrna", C=0.1))
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

        assert pipeline.named_steps["selection"].get_support().any()



def test_all_three_feature_groups_fit():
    """All demographic, lncRNA and combined pipelines should fit."""
    from scripts.ml.data_builder import build_dataset

    data = build_dataset("data", "GSE63061")

    groups = {
        "demographic": data["X_demographic"],
        "lncrna": data["X_lncrna"],
        "combined": data["X_combined"],
    }

    for group, X in groups.items():
        pipeline = make_lasso_selector(group, C=0.1)
        pipeline.fit(X, data["y"])

        assert pipeline.named_steps["selection"].get_support().any()
