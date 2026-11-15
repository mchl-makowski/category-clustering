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
