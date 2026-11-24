"""Stage 15 candidate pipelines. Each fit learns its own Stage 14 transformations."""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from scripts.ml.preprocessing import make_preprocessing_pipeline

SEED = 42
FEATURE_SETS = ("age_sex_only", "lncrna_only", "combined")
GROUPS = dict(zip(FEATURE_SETS, ("demographic", "lncrna", "combined")))


def build_model_registry(n_features, feature_set="lncrna_only"):
    """Return candidates in deterministic tie-break order for the discovery study."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    k = sorted({min(n_features, n) for n in (5, 20)})
    def pipeline(model, select=False):
        steps = make_preprocessing_pipeline(GROUPS[feature_set]).steps
        if select:
            steps.append(("select", SelectKBest(f_classif, k=k[0])))
        return Pipeline(steps + [("model", model)])

    return {
        "logistic_baseline": (pipeline(LogisticRegression(C=float("inf"), max_iter=4000)), {}),
        "logistic_l1": (pipeline(LogisticRegression(solver="liblinear", l1_ratio=1,
                        random_state=SEED, max_iter=4000)), {"model__C":[0.05, 0.2, 1.0]}),
        "elastic_net": (pipeline(LogisticRegression(solver="saga", l1_ratio=0.5, tol=1e-3,
                        random_state=SEED, max_iter=6000)),
                        {"model__C":[0.05, 0.2], "model__l1_ratio":[0.25, 0.75]}),
        "random_forest": (pipeline(RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                          class_weight="balanced", random_state=SEED, n_jobs=1), True),
                          {"select__k":k, "model__max_depth":[3, 5]}),
        "rbf_svm": (pipeline(CalibratedClassifierCV(SVC(kernel="rbf", class_weight="balanced"),
                         ensemble=False, cv=3, method="sigmoid"), True),
                         {"select__k":k, "model__estimator__C":[0.1, 1.0]}),
    }
