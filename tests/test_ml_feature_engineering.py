import numpy as np
import pandas as pd
import pytest

from scripts.ml.data_builder import build_dataset
from scripts.ml.feature_selection import make_lasso_selector, selected_feature_names, selection_stability
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold


@pytest.fixture
def dataset(tmp_path):
    """Stage 10 file layout with synthetic inputs for algorithm tests only."""
    root = tmp_path / "data"

    matrix_dir = root / "processed" / "stage10"
    metadata_dir = root / "interim" / "GSE63061"
    annotation_dir = root / "annotations"

    matrix_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)

    sample_ids = [f"S{i}" for i in range(10)]
    X = pd.DataFrame(
        [[float(i % 2), 1.0] for i in range(10)],
        index=sample_ids,
        columns=["P1", "P2"],
    )
    X.to_parquet(
        matrix_dir / "GSE63061_validated_lncrna.parquet"
    )

    metadata = pd.DataFrame({
        "sample_id": sample_ids,
        "dataset_id": ["GSE63061"] * 10,
        "diagnosis": ["AD", "Control"] * 4 + ["MCI", "Unknown"],
        "age": [70, 65, None, 68, 70, 66, 200, None, 68, 72],
        "sex": [" F ", "M", None, "  ", "F", "F", "unseen", " ", "F", "M"],
        "batch": ["A"] * 10,
        "tissue": ["blood"] * 10,
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

    return build_dataset(root, "GSE63061")


def test_build_dataset_excludes_non_binary_diagnoses(dataset):
    result = dataset

    assert result["X_lncrna"].shape == (8, 2)
    assert result["X_combined"].shape == (8, 4)
    assert result["y"].to_dict() == {f"S{i}": 1 - i % 2 for i in range(8)}
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

    X["P0"] = y * 5.0
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for train_idx, _ in cv.split(X, y):
        pipeline = clone(make_lasso_selector("lncrna", C=0.1))
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])

        assert pipeline.named_steps["selection"].get_support().any()



@pytest.mark.parametrize("group", ["demographic", "lncrna", "combined"])
def test_all_three_feature_groups_fit(dataset, group):
    X = dataset[f"X_{group}"]
    y = dataset["y"]
    model = make_lasso_selector(group, C=100).fit(X.iloc[:6], y.iloc[:6])
    transformed = model.transform(X.iloc[6:])
    assert transformed.shape == (2, len(selected_feature_names(model)))
    assert np.isfinite(transformed).all()
    if group != "lncrna":
        assert X.sex.iloc[0] == "F"
        assert X.sex.iloc[[2, 3, 7]].isna().all()
        prep = model.named_steps["preprocessing"]
        columns = prep.named_steps["impute_encode"]
        cat = columns.named_transformers_["categorical"]
        assert cat.named_steps["imputer"].statistics_.tolist() == ["F"]
        assert cat.named_steps["encoder"].categories_[0].tolist() == ["F", "M"]
        numeric = columns.named_transformers_["numeric"].named_steps["imputer"]
        assert numeric.statistics_[-1] == 68
        scaled_inputs = prep[:-1].transform(X.iloc[:6])
        np.testing.assert_allclose(prep.named_steps["scale"].mean_, scaled_inputs.mean(axis=0))


def test_zero_selection_and_downstream_failure(dataset):
    X, y = dataset["X_lncrna"], dataset["y"]
    selector = make_lasso_selector(C=1e-10).fit(X, y)
    selection = selector.named_steps["selection"]
    assert not selection.estimator_.coef_.any()
    assert not selection.get_support().any()
    assert selected_feature_names(selector) == []
    for call in [lambda: selector.transform(X),
                 lambda: make_pipeline(clone(selector), LogisticRegression()).fit(X, y)]:
        with pytest.raises(ValueError, match="LASSO selected zero features"):
            call()
    stability, folds = selection_stability(X, y, selector, StratifiedKFold(2))
    assert stability.empty and "feature_id" in stability
    assert folds.selected_feature_count.tolist() == [0, 0]


def test_selected_names_match_transformed_columns(dataset):
    X, y = dataset["X_combined"], dataset["y"]
    model = make_lasso_selector("combined", C=2).fit(X, y)
    prep = model.named_steps["preprocessing"]
    names = prep.get_feature_names_out().tolist()
    selected = selected_feature_names(model)
    assert "P1" in selected and "P2" not in names
    expected = prep.transform(X)[:, [names.index(name) for name in selected]]
    np.testing.assert_allclose(model.transform(X), expected)


def test_stability_fits_training_rows_and_labels_only(dataset, monkeypatch):
    X, y = dataset["X_combined"].copy(), dataset["y"]
    X["heldout_only"] = [0] * 6 + [1, 2]
    observed = []
    original_fit = LogisticRegression.fit

    def record_fit(estimator, matrix, labels, **kwargs):
        observed.append((matrix.copy(), labels.copy()))
        return original_fit(estimator, matrix, labels, **kwargs)

    class FixedSplit:
        def split(self, X, y):
            yield np.arange(6), np.arange(6, 8)

    monkeypatch.setattr(LogisticRegression, "fit", record_fit)
    selector = make_lasso_selector("combined", C=10)
    first, _ = selection_stability(X, y, selector, FixedSplit())
    X.iloc[6:, X.columns.get_loc("age")] = -1000
    changed = y.copy()
    changed.iloc[6:] = 1 - changed.iloc[6:]
    second, _ = selection_stability(X, changed, selector, FixedSplit())
    pd.testing.assert_frame_equal(first, second)
    assert len(observed) == 2
    for matrix, labels in observed:
        pd.testing.assert_series_equal(labels, y.iloc[:6])
        # P1, age, sex_F and sex_M survive training-fold variance filtering.
        assert matrix.shape == (6, 4)
        np.testing.assert_allclose(matrix.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_array_equal(observed[0][0], observed[1][0])
