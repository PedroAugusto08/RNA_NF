# Configuracoes centrais do projeto.

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
PREPROCESSING_DIR = TABLES_DIR / "preprocessing"
EXPERIMENTS_DIR = TABLES_DIR / "experiments"


TRAIN_SIZE = 0.60
VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20
DEFAULT_RANDOM_SEED = 42
DEFAULT_EXPERIMENT_RUNS = 3
RECOMMENDED_EXPERIMENT_RUNS = 21


DATASET_CONFIGS = {
    "breast_cancer": {
        "display_name": "Breast Cancer Wisconsin",
        "filename": "breast-cancer.csv",
        "task_type": "binary_classification",
        "target_column": "diagnosis",
        "drop_columns": ["id"],
        "target_labels_known": False,
    },
    "diabetes": {
        "display_name": "Pima Indians Diabetes",
        "filename": "diabetes.csv",
        "task_type": "binary_classification",
        "target_column": "Outcome",
        "drop_columns": [],
        "target_labels_known": True,
    },
    "faults": {
        "display_name": "Steel Plates Faults",
        "filename": "faults.csv",
        "task_type": "multiclass_classification",
        "target_column": "fault_type",
        "one_hot_target_columns": [
            "Pastry",
            "Z_Scratch",
            "K_Scatch",
            "Stains",
            "Dirtiness",
            "Bumps",
            "Other_Faults",
        ],
        "drop_columns": [],
        "target_labels_known": True,
    },
    "robot_navigation": {
        "display_name": "Wall-Following Robot Navigation",
        "filename": "Robo.csv",
        "task_type": "multiclass_classification",
        "target_column": "Class",
        "drop_columns": ["id"],
        "target_labels_known": False,
    },
}


DATASET_ALIASES = {
    "breast-cancer": "breast_cancer",
    "breast_cancer": "breast_cancer",
    "cancer": "breast_cancer",
    "diabetes": "diabetes",
    "pima": "diabetes",
    "faults": "faults",
    "steel": "faults",
    "steel_plates_faults": "faults",
    "robot": "robot_navigation",
    "robot_navigation": "robot_navigation",
    "wall_following_robot": "robot_navigation",
}
