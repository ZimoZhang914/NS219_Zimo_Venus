"""
Typed configuration object for the pipeline.

A YAML file is parsed into a nested dict and bound to these dataclasses,
so editor autocompletion and type-checking work in VS Code.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ----- sub-configs ----------------------------------------------------------


@dataclass
class DataConfig:
    h5ad_path: Optional[str] = None
    tenx_dir: Optional[str] = None
    celltype_col: str = "cell_type"
    microglia_labels: List[str] = field(
        default_factory=lambda: ["Microglia", "Macrophage", "PVM", "Perivascular Macrophage"]
    )
    donor_col: str = "donor_id"
    condition_col: str = "diagnosis"
    ad_label: str = "AD"
    control_label: str = "Control"


@dataclass
class QCConfig:
    min_genes_per_cell: int = 200
    min_cells_per_gene: int = 3
    max_mt_percent: float = 10.0
    max_counts: int = 50_000


@dataclass
class PreprocessConfig:
    n_top_genes: int = 3000
    target_sum: float = 1e4
    scale: bool = False
    batch_correction: str = "harmony"   # "none" | "harmony"
    batch_key: str = "donor_id"


@dataclass
class VAEConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    latent_dim: int = 32
    dropout: float = 0.1
    likelihood: str = "poisson"          # "poisson" | "mse"
    beta: float = 1.0
    warmup_epochs: int = 10
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 15
    val_fraction: float = 0.1


@dataclass
class ClusteringConfig:
    resolutions: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    )
    n_neighbors: int = 15


@dataclass
class ClassifierConfig:
    reference_label_col: str = "cell_state"
    model: str = "xgboost"               # "xgboost" | "logistic" | "svc"
    use_optuna: bool = True
    optuna_trials: int = 50
    test_fraction: float = 0.2
    use_smote: bool = True
    confidence_threshold: float = 0.7


@dataclass
class DAMConfig:
    extra_signatures: Dict[str, List[str]] = field(default_factory=dict)
    test: str = "wilcoxon"


@dataclass
class EvalConfig:
    make_umap: bool = True
    umap_color_by: List[str] = field(
        default_factory=lambda: ["predicted_state", "DAM_score", "condition"]
    )
    save_h5ad: bool = True


# ----- top-level ------------------------------------------------------------


@dataclass
class Config:
    run_name: str = "vineseq_microglia_v1"
    output_dir: str = "results/vineseq_microglia_v1"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    dam: DAMConfig = field(default_factory=DAMConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ----- helpers ---------------------------------------------------------

    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


# ----- loading --------------------------------------------------------------


def _build(cls, raw: Optional[Dict[str, Any]]):
    """Recursively build a dataclass from a (possibly partial) dict."""
    if raw is None:
        return cls()
    kwargs = {}
    for f in cls.__dataclass_fields__.values():
        if f.name in raw:
            value = raw[f.name]
            # nested dataclass?
            if hasattr(f.type, "__dataclass_fields__"):
                kwargs[f.name] = _build(f.type, value)
            else:
                kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load a YAML file into a Config object, falling back to defaults."""
    raw = yaml.safe_load(Path(path).read_text()) or {}

    cfg = Config(
        run_name=raw.get("run_name", "vineseq_microglia_v1"),
        output_dir=raw.get("output_dir", "results/vineseq_microglia_v1"),
        seed=raw.get("seed", 42),
        data=_build(DataConfig, raw.get("data")),
        qc=_build(QCConfig, raw.get("qc")),
        preprocess=_build(PreprocessConfig, raw.get("preprocess")),
        vae=_build(VAEConfig, raw.get("vae")),
        clustering=_build(ClusteringConfig, raw.get("clustering")),
        classifier=_build(ClassifierConfig, raw.get("classifier")),
        dam=_build(DAMConfig, raw.get("dam")),
        eval=_build(EvalConfig, raw.get("eval")),
    )
    return cfg
