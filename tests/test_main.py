import pytest

from projeto_integrador import __version__
from projeto_integrador.main import main


def test_versao() -> None:
    assert __version__ == "0.1.0"


def test_main(capsys: pytest.CaptureFixture[str]) -> None:
    main()
    assert capsys.readouterr().out.strip() == "Projeto integrador pronto."
