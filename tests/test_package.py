from pathlib import Path

import crossjit_generator


def test_package_version_is_exposed():
    assert crossjit_generator.__version__ == "0.2.0"


def test_generic_source_has_no_numpy_implementation_dependencies():
    root = Path(crossjit_generator.__file__).parent
    forbidden = ("PyArrayObject", "NPY_", "numpy/arrayobject", "ufuncobject")
    source = "\n".join(
        path.read_text()
        for path in sorted(root.rglob("*.py"))
    )
    for token in forbidden:
        assert token not in source
