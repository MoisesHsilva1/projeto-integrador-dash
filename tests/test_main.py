from credit_card_fraud import __version__
from credit_card_fraud.main import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_main(capsys) -> None:
    main()
    expected = "Credit card fraud detection project ready."
    assert capsys.readouterr().out.strip() == expected
