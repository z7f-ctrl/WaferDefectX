"""Project path helpers — always resolve from this file, never from cwd."""
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PYTHON_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_SYNTHETIC = DATA_DIR / "synthetic"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
CPP_DIR = PROJECT_ROOT / "cpp"
CPP_BUILD_DIR = CPP_DIR / "build"
CPP_EXECUTABLE = CPP_BUILD_DIR / "WaferDefectX_Run"

# Feature / model contract (shared with export + classifier)
FEATURE_DIM = 7
FEATURE_VERSION = "1.0"
FEATURE_NAMES = [
    "area",
    "perimeter",
    "aspect_ratio",
    "rectangularity",
    "circularity",
    "mean_intensity",
    "std_intensity",
]
ONNX_INPUT_NAME = "float_input"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
