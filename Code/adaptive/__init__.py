
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "adaptive" / "models" / "Refine"))
sys.path.insert(0, str(ROOT / "adaptive" / "models" / "Deconv"))
sys.path.insert(0, str(ROOT / "AMP"))


try:
    from torchvision.transforms import functional as _tvf

    if "torchvision.transforms.functional_tensor" not in sys.modules:
        sys.modules["torchvision.transforms.functional_tensor"] = types.SimpleNamespace(
            rgb_to_grayscale=_tvf.rgb_to_grayscale
        )
except Exception:
    pass
