"""CLI data selection works without distributing the private input data."""
import json

import pytest

from bayesian_ring_ssr import cli


@pytest.mark.parametrize("explicit_input", [False, True])
def test_audit_data_environment_default_and_explicit_override(monkeypatch, capsys, tmp_path, explicit_input):
    configured = tmp_path / "private data" / "DM_16_aktive_Ringe.pkl"
    override = tmp_path / "other data" / "DM_16_aktive_Ringe.pkl"
    monkeypatch.setenv("BAYESIAN_SSR_DATA", str(configured))
    requested = []

    def audit(path):
        requested.append(path)
        return {"input_checked": path}

    monkeypatch.setattr(cli, "audit_data", audit)
    arguments = ["audit-data"]
    if explicit_input:
        arguments += ["--input", str(override)]

    cli.main(arguments)

    expected = str(override if explicit_input else configured)
    assert requested == [expected]
    assert json.loads(capsys.readouterr().out) == {"input_checked": expected}
