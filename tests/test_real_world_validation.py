from pathlib import Path

import pandas as pd

from scripts.experiments.amiga_exp.real_world_validation import (
    Selector,
    build_reported_markdown,
    build_reported_source_support_table,
)


def test_reported_source_support_table_uses_only_paper_sources(tmp_path: Path):
    network_dir = tmp_path / "networks"
    network_dir.mkdir()
    network_path = network_dir / "AMIGA_top1_edges.csv"
    pd.DataFrame(
        [
            {"edge_rank": 1, "source": "TP53", "target": "ESR1", "confidence": 0.9, "source_norm": "TP53", "target_norm": "ESR1"},
            {"edge_rank": 2, "source": "FOXA1", "target": "GATA3", "confidence": 0.8, "source_norm": "FOXA1", "target_norm": "GATA3"},
            {"edge_rank": 3, "source": "MYC", "target": "ESR1", "confidence": 0.7, "source_norm": "MYC", "target_norm": "ESR1"},
        ]
    ).to_csv(network_path, index=False)
    evidence = pd.DataFrame(
        [
            {"source": "TP53", "target": "ESR1", "resource": "OmniPath_CollecTRI", "evidence_type": "tf_target"},
            {"source": "FOXA1", "target": "GATA3", "resource": "OmniPath_DoRothEA", "evidence_type": "tf_target"},
            {"source": "FOXA1", "target": "GATA3", "resource": "TRRUST_v2", "evidence_type": "tf_target"},
            {"source": "MYC", "target": "ESR1", "resource": "Enrichr_JASPAR_2025", "evidence_type": "motif"},
            {"source": "TP53", "target": "ESR1", "resource": "CistromeCancer_BRCA_COR", "evidence_type": "brca_cor"},
            {"source": "MYC", "target": "GATA3", "resource": "TFLink", "evidence_type": "excluded"},
        ]
    )
    selectors = [
        Selector(
            selector_id="AMIGA_top1",
            method_family="learned_post_pareto",
            selected_index=0,
            selected_item_id=147,
            selector_score=1.0,
        )
    ]

    table = build_reported_source_support_table(
        selectors=selectors,
        network_paths={"AMIGA_top1": network_path},
        evidence=evidence,
    )

    assert list(table["method"]) == ["AMIGA"]
    assert table.loc[0, "collectri_tf_target_edges"] == 1
    assert table.loc[0, "collectri_tf_source_edges"] == 1
    assert table.loc[0, "dorothea_tf_target_edges"] == 1
    assert table.loc[0, "trrust_v2_tf_target_edges"] == 1
    assert table.loc[0, "jaspar_pwm_tf_target_edges"] == 1
    assert table.loc[0, "cistrome_brca_cor_tf_target_edges"] == 1
    assert not any("tflink" in column.lower() for column in table.columns)

    markdown = build_reported_markdown(table)
    assert "Top1 selectors only" in markdown
    assert "Top5" not in markdown
    assert "coexpression" not in markdown.lower()
