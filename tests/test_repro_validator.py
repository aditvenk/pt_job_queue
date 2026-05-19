from __future__ import annotations

from ptq.evaluator.repro_validator import infer_repro_source, validate_repro_presence


def test_infers_issue_repro_from_filename():
    assert infer_repro_source(issue_number=174923, repro_filename="repro_174923.py") == "from_issue"


def test_infers_generated_repro_from_filename():
    assert (
        infer_repro_source(
            issue_number=174923,
            repro_filename="repro_174923_generated.py",
        )
        == "generated"
    )


def test_missing_repro_blocks_evaluation():
    check = validate_repro_presence(
        issue_number=174923,
        repro_filename="",
        repro_script="",
    )
    assert check.blocks_evaluation is True
    assert check.comments[0].severity == "blocking"
