from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_retired_legacy_experiment_scripts_are_removed():
    retired_paths = [
        REPO_ROOT / "scripts" / "experiments" / "phase1" / "01_cv_model_screen.sh",
        REPO_ROOT / "scripts" / "experiments" / "phase1" / "03_plot_phase1_results.sh",
        REPO_ROOT / "scripts" / "experiments" / "phase1" / "02_cv_hyperparameter_tuning.sh",
        REPO_ROOT / "scripts" / "analysis" / "select_finalists.py",
        REPO_ROOT / "scripts" / "experiments" / "phase2" / "01_cv_ablation.sh",
        REPO_ROOT / "scripts" / "experiments" / "phase2" / "02_plot_ablation.sh",
        REPO_ROOT / "scripts" / "experiments" / "phase3" / "01_decision_baselines.sh",
        REPO_ROOT / "scripts" / "experiments" / "phase3" / "02_plot_decision_baselines.sh",
        REPO_ROOT / "scripts" / "analysis" / "build_decision_baselines.py",
        REPO_ROOT / "scripts" / "analysis" / "experiment_stat_tests.py",
    ]

    assert all(not path.exists() for path in retired_paths)
    assert not any((REPO_ROOT / "scripts" / "experiments").glob("phase*"))
    assert not (REPO_ROOT / "scripts" / "analysis").exists()


def test_readme_points_to_current_experimental_results_layout():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "amiga summarize-cv <cv_reports_dir>/*/cv_report.json" in readme
    assert "experiments/BIO-INSIGHT" not in readme
    assert "phase1/01_model_screen/cv_results" not in readme


def test_implementation_plan_documents_are_not_kept_in_versioned_docs():
    forbidden_plan_docs = [
        REPO_ROOT / "docs" / "experiments" / "implementation_plan.md",
        REPO_ROOT / "docs" / "experiments" / "reporting_plot_plan.md",
        REPO_ROOT / "docs" / "experiments" / "phase4_decision_baselines_plan.md",
    ]

    assert all(not path.exists() for path in forbidden_plan_docs)


def test_versioned_markdown_avoids_ignored_local_paths():
    markdown_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "experiments.md",
        REPO_ROOT / "docs" / "experiments" / "design.md",
    ]
    forbidden_fragments = [
        "experiments/BIO-INSIGHT",
        "experiments/MO-GENECI",
        "paper/",
        "state_of_art",
        "deep-research",
        "data/training",
        "data/new_front",
        "data/expression",
        "data/GRN",
    ]

    for path in markdown_paths:
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in text, f"{fragment!r} found in {path}"
