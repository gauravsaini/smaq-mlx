"""Install SMAQ hooks into the currently installed mlx-lm.

Adds a small auto-detect block to mlx_lm/__init__.py that activates
SMAQ when the SMAQ_ENABLED=1 environment variable is set.

Usage:
    python -m smaq_mlx.install          # install the hook
    python -m smaq_mlx.install --revert # remove the hook
"""

import argparse
import importlib
import sys
from pathlib import Path

SMAQ_HOOK_START = "# --- SMAQ HOOK START ---"
SMAQ_HOOK_END = "# --- SMAQ HOOK END ---"

SMAQ_HOOK = f"""\n{SMAQ_HOOK_START}
# Auto-activate SMAQ KV cache compression when SMAQ_ENABLED=1
import os as _os
if _os.environ.get("SMAQ_ENABLED", "0") == "1":
    try:
        from smaq_mlx.patch import apply as _smaq_apply
        _smaq_apply()
    except ImportError:
        pass
    except Exception as _e:
        import warnings as _w
        _w.warn(f"SMAQ patch failed: {{_e}}")
{SMAQ_HOOK_END}
"""


def _find_mlx_lm_init() -> Path:
    """Locate the installed mlx_lm __init__.py."""
    spec = importlib.util.find_spec("mlx_lm")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "mlx-lm is not installed in the current Python environment. "
            "Install it first: pip install mlx-lm"
        )
    return Path(spec.origin)


def install():
    """Add the SMAQ hook to mlx_lm/__init__.py."""
    init_path = _find_mlx_lm_init()
    content = init_path.read_text()

    if SMAQ_HOOK_START in content:
        print(f"[SMAQ] Hook already installed in {init_path}")
        return

    content += SMAQ_HOOK
    init_path.write_text(content)
    print(f"[SMAQ] ✅ Hook installed in {init_path}")
    print(f"[SMAQ] Usage:")
    print(f"  SMAQ_ENABLED=1 python -m mlx_lm.generate --model <model> --prompt 'hello'")
    print(f"  SMAQ_ENABLED=1 python -m mlx_lm.server --model <model>")
    print(f"")
    print(f"  Environment variables:")
    print(f"    SMAQ_ENABLED=1         Enable SMAQ KV cache (required)")
    print(f"    SMAQ_KEY_BITS=4        Key quantization bits (default: 4)")
    print(f"    SMAQ_VALUE_BITS=4      Value quantization bits (default: 4)")


def revert():
    """Remove the SMAQ hook from mlx_lm/__init__.py."""
    init_path = _find_mlx_lm_init()
    content = init_path.read_text()

    if SMAQ_HOOK_START not in content:
        print(f"[SMAQ] No hook found in {init_path}")
        return

    start = content.index(SMAQ_HOOK_START)
    # Go back to the newline before the marker
    while start > 0 and content[start - 1] == "\n":
        start -= 1
    end = content.index(SMAQ_HOOK_END) + len(SMAQ_HOOK_END)
    # Include trailing newline
    if end < len(content) and content[end] == "\n":
        end += 1

    content = content[:start] + content[end:]
    init_path.write_text(content)
    print(f"[SMAQ] ✅ Hook removed from {init_path}")


def main():
    parser = argparse.ArgumentParser(description="Install/remove SMAQ hooks in mlx-lm")
    parser.add_argument("--revert", action="store_true", help="Remove the SMAQ hook")
    args = parser.parse_args()

    if args.revert:
        revert()
    else:
        install()


if __name__ == "__main__":
    main()
