from __future__ import annotations

import json
import subprocess
import sys


def test_import_geofarmai_does_not_load_application_or_llm_modules(repository_root):
    forbidden_prefixes = [
        "ollama",
        "langchain",
        "streamlit",
        "fastapi",
        "openai",
        "rpy2",
    ]
    script = f"""
import json
import sys
import geofarmai

prefixes = {forbidden_prefixes!r}
loaded = sorted(
    name for name in sys.modules
    if any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes)
)
print(json.dumps({{'loaded': loaded, 'package_file': geofarmai.__file__}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    result = json.loads(completed.stdout)
    assert result["loaded"] == []
    assert result["package_file"] == str(repository_root / "geofarmai" / "__init__.py")
