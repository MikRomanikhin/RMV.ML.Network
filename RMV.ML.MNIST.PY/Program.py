import json
# import sys
from pathlib import Path

from Multi_Layer_Net import AppSettings, ActivationType, OptimizerType
from Parser import Parser
from Controller import Controller


# ---------------------------------------------------------------------------
# ConfigManager  —  mirrors C# ConfigManager.GetRoot<AppSettings>()
# ---------------------------------------------------------------------------
def load_settings(path: str = "appsettings.json") -> AppSettings:
    """
    Loads AppSettings from a JSON file.
    Mirrors C# ConfigManager.GetRoot<AppSettings>().
    Raises if the file is not found (mirrors ?? throw new Exception(...)).
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Failed to load configuration: '{path}' not found.")

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    return AppSettings(
        input           = data.get("input",          784),
        hidden          = data.get("hidden",          [100]),
        output          = data.get("output",          10),
        activation      = ActivationType(data.get("activation", "Relu")),
        optimizer       = OptimizerType(data.get("optimizer",   "SGD")),
        decay           = data.get("decay",           0.0),
        rate            = data.get("rate",            0.01),
        momentum        = data.get("momentum",        0.9),
        iterations      = data.get("iterations",      10000),
        batch           = data.get("batch",           100),
        epoch           = data.get("epoch",           10),
        print_interval  = data.get("print_interval",  100),
        stagnation      = data.get("stagnation",      100),
        weight_init_std = data.get("weight_init_std", 0.01),
        error_path      = data.get("error_path",      "errors.txt"),
        index_path      = data.get("index_path",      "indexes.txt"),
    )


# ---------------------------------------------------------------------------
# Entry point  —  mirrors C# Program.Main()
# ---------------------------------------------------------------------------
def main() -> None:
    settings = load_settings("appsettings.json")

    parser = Parser(settings.output)

    print("Reading datasets...")
    train_lines = Path(settings.train_path).read_text(encoding="utf-8").splitlines()
    test_lines  = Path(settings.test_path ).read_text(encoding="utf-8").splitlines()

    # Strip empty lines — mirrors C# ReadAllLines (skips blank lines implicitly)
    train_lines = [l for l in train_lines if l.strip()]
    test_lines  = [l for l in test_lines  if l.strip()]

    train_set = parser.run(train_lines)
    test_set  = parser.run(test_lines)

    controller = Controller(train_set, test_set, settings)
    controller.run_ml()


if __name__ == "__main__":
    main()