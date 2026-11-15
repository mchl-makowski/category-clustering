import re

from problem_clustering.cli import main


def test_run_method_a_on_fixture_with_fake_backend(tmp_path, fixture_path):
    code = main([
        "run", "--method", "a", "--input", str(fixture_path), "--out-dir", str(tmp_path),
        "--set", "backend.kind=fake", "--set", "method_a.distance_threshold=0.6", "--no-judge",
    ])
    assert code == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 2
    assert re.fullmatch(r"\d{6}_\d{4}_a\.txt", names[0])
    assert re.fullmatch(r"\d{6}_\d{4}_a\.xlsx", names[1])


def _judge_rows(xlsx):
    import pandas as pd

    return len(pd.read_excel(xlsx, sheet_name="Judge"))


def test_run_judges_by_default_and_no_judge_skips_it_then_compare(tmp_path, fixture_path):
    base = ["run", "--method", "a", "--input", str(fixture_path), "--set", "backend.kind=fake",
            "--set", "method_a.distance_threshold=0.6"]
    assert main(base + ["--out-dir", str(tmp_path / "judged")]) == 0
    assert main(base + ["--out-dir", str(tmp_path / "plain"), "--no-judge"]) == 0
    judged = next((tmp_path / "judged").glob("*.xlsx"))
    plain = next((tmp_path / "plain").glob("*.xlsx"))
    assert _judge_rows(judged) > 0
    assert _judge_rows(plain) == 0

    assert main(["compare", str(judged), str(plain), "--out-dir", str(tmp_path / "cmp")]) == 0
    out = list((tmp_path / "cmp").iterdir())
    assert len(out) == 1 and re.fullmatch(r"\d{6}_\d{4}_comparison\.xlsx", out[0].name)
