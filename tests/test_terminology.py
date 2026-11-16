"""The vocabulary of the contract, kept.

``docs/19_Contract.md`` is the reference for what things are called. A name it
lists as replaced must not come back: not in the code, not in the pages, not in
a notebook cell. That page alone may name them (it says what replaced what),
and the tests that pin a refusal may too.
"""
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: A replaced name, and what to write instead.
RETIRED = {
    r"\bInteractiveKalmanFilter\b": "route(models, Pi)",
    r"\bFilters::IKF\b": "Models::Router on Engines::Interacting",
    r"\bIKF\b": "route / the interacting engine (IMM in the literature)",
    r"\bikf_bank\b": "regime_bank",
    r"\bmu_log\b": "mode_probabilities",
    r"\bnis_log\b": "nis_history",
    r"\bCompositeKalmanModel\b": "compose / Composite",
    r"\bKalmanObservableReadout\b": "KalmanReadout (STACKED, SUMMED)",
    r"\bobs_mode\b": "readout=",
    r"\bAugmentedKalmanModel\b": "joint / Joint",
    r"\bNativeTwin\b": "NativeCounterpart",
    r"\bKALMAN_NATIVE_TWINS\b": "KALMAN_NATIVE_COUNTERPARTS",
    r"\binitial_state_vector\b": "initial_state(x0, P0)",
    r"\binitial_state_covariance\b": "initial_state(x0, P0)",
    r"\bget_initial_state\w*": "initial_state()",
    r"\bsample_initial_state\b": "initial_state(samples=N)",
    r"\bmodel\.initial_ensemble\(": "model.initial_state(samples=N); initial_ensemble is the ensemble engine's own",
    r"\bsensor_id\b": "sensor",
    r"\bremove_sensor\b": "a model is made with its sensors (add_sensor gives a new model)",
    r"\bclear_sensors\b": "a model is made with its sensors",
    r"\.observe\(": "h (given many states), or the ensemble engine's project",
}

#: Where a retired name may appear: the page that records the replacements, this
#: file, and the tests that check the old names are refused.
ALLOWED = {"docs/19_Contract.md", "tests/test_terminology.py", "tests/test_structure.py", "tests/test_add_sensor.py"}

TEXT = (".py", ".md", ".h", ".cc", ".txt", ".toml", ".yaml", ".yml", ".cmake")
SKIP = ("var", ".git", "__pycache__", "node_modules", ".ruff_cache")


def _files():
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(part in SKIP for part in rel.parts) or rel.as_posix() in ALLOWED:
            continue
        if path.suffix in TEXT or path.suffix == ".ipynb":
            yield path, rel.as_posix()


def _text(path):
    """What a person wrote in the file: for a notebook the cell sources, not the stored outputs."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix != ".ipynb":
        return raw
    try:
        cells = json.loads(raw).get("cells", [])
    except json.JSONDecodeError:
        return raw
    return "\n".join("".join(cell.get("source", [])) for cell in cells)


class TestTerminology(unittest.TestCase):

    def test_replaced_names_are_gone(self):
        patterns = {re.compile(pattern): instead for pattern, instead in RETIRED.items()}
        found = []
        for path, rel in _files():
            text = _text(path)
            for pattern, instead in patterns.items():
                for match in pattern.finditer(text):
                    line = text.count("\n", 0, match.start()) + 1
                    found.append(f"{rel}:{line}: {match.group(0)!r} -> {instead}")
        self.assertEqual(found, [], "\n" + "\n".join(found[:40]))

    def test_the_contract_page_names_what_replaced_them(self):
        page = (ROOT / "docs" / "19_Contract.md").read_text(encoding="utf-8")
        for word in ("route", "compose", "joint", "add_sensor", "add_instrument", "lag", "covariance_mask",
                     "initial_state", "KalmanSensor", "KalmanInstrument", "step", "sense"):
            self.assertIn(word, page, word)
        self.assertNotRegex(page, r"\| step 1\d \|", "an operation is still marked as planned")

    def test_every_operation_is_a_method_a_function_and_a_class(self):
        import sys
        sys.path.insert(0, str(ROOT))
        import modules.framework as framework
        from modules.framework.model import KalmanModel
        for verb, noun in (("joint", "Joint"), ("add_sensor", "Fusion"), ("lag", "Lag"), ("compose", "Composite"),
                           ("route", "Router")):
            self.assertTrue(callable(getattr(KalmanModel, verb, None)), f"KalmanModel.{verb}")
            self.assertTrue(callable(getattr(framework, verb, None)), f"framework.{verb}")
            self.assertTrue(isinstance(getattr(framework, noun, None), type), f"framework.{noun}")
        self.assertTrue(callable(getattr(framework, "add_instrument", None)))


if __name__ == "__main__":
    unittest.main()
