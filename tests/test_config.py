import pytest

from labelrag import config


@pytest.fixture
def no_env_key(monkeypatch):
    monkeypatch.delenv("CO_API_KEY", raising=False)


def test_missing_key_explains_how_to_set_it(no_env_key, tmp_path):
    with pytest.raises(config.MissingAPIKeyError, match=r"Copy \.env\.example to \.env"):
        config.api_key(env_file=tmp_path / "absent.env")


def test_blank_key_counts_as_missing(no_env_key, tmp_path):
    env = tmp_path / ".env"
    env.write_text("CO_API_KEY=   \n", encoding="utf-8")
    with pytest.raises(config.MissingAPIKeyError):
        config.api_key(env_file=env)


def test_key_is_read_from_env_file(no_env_key, tmp_path):
    env = tmp_path / ".env"
    env.write_text("CO_API_KEY=from-file\n", encoding="utf-8")
    assert config.api_key(env_file=env) == "from-file"


def test_environment_variable_wins_over_env_file(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("CO_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("CO_API_KEY", "from-environment")
    assert config.api_key(env_file=env) == "from-environment"


def test_main_reports_a_missing_key(no_env_key, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent.env")
    assert config.main() == 1
    assert "CO_API_KEY: missing" in capsys.readouterr().err


def test_main_never_prints_the_key(monkeypatch, capsys):
    monkeypatch.setenv("CO_API_KEY", "not-a-real-key-123")
    assert config.main() == 0
    output = capsys.readouterr()
    assert "not-a-real-key-123" not in output.out + output.err
