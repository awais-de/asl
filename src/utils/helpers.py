import importlib
import json
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass
import subprocess
import sys
from typing import Literal
import re
import contractions

logger = logging.getLogger(__name__)

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
RUN_INFO_PATH = PROJECT_ROOT / "run_info.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"  # All artifacts stored here

@dataclass
class Artifact:
    """
    Represents a generated or required file (artifact) in the system.
    """
    name: str  # e.g., "aslg_pc12_clean.jsonl"
    type: Literal["json", "jsonl", "txt", "pt", "csv", "other"]
    run_id: int
    use_run_folder: bool = True
    timestamp: str = None  # auto-generated if None

    def build_name(self) -> str:
        """
        Constructs the artifact filename with run_id and timestamp embedded.
        Example: run42_20250813_103412_aslg_pc12_clean.jsonl
        """
        base_path = Path(self.name)
        stem = base_path.stem
        suffix = base_path.suffix

        if self.timestamp is None:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"run{self.run_id}_{self.timestamp}_{stem}{suffix}"

    def get_path(self) -> Path:
        """
        Returns the full path to the artifact in the artifacts directory.
        """
        if self.use_run_folder:
            run_folder = ARTIFACTS_DIR / f"run{self.run_id}"
            run_folder.mkdir(parents=True, exist_ok=True)
        else:
            run_folder = ARTIFACTS_DIR
            run_folder.mkdir(parents=True, exist_ok=True)

        return run_folder / self.build_name()


# ------------------- Package Install Helper Functions -------------------

def is_installed(pkg_name):
    """Check if a package is already installed."""
    spec = importlib.util.find_spec(pkg_name)
    return spec is not None

def install_package(pkg_name):
    """Install a Python package quietly via pip."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def uninstall_package(pkg_name):
    """Uninstall a Python package quietly via pip."""
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", pkg_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

# ------------------- Run Info Management -------------------

def read_run_info() -> dict:
    if RUN_INFO_PATH.exists():
        with open(RUN_INFO_PATH, "r") as f:
            return json.load(f)
    logger.warning(f"Run info file {RUN_INFO_PATH} not found, returning empty dict.")
    return {}


def write_run_info(data: dict):
    with open(RUN_INFO_PATH, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Updated run info written to {RUN_INFO_PATH}")


def get_next_run_id() -> int:
    data = read_run_info()
    last_id = data.get("last_run_id", 0)
    new_id = last_id + 1
    data["last_run_id"] = new_id
    if "runs" not in data:
        data["runs"] = {}
    write_run_info(data)
    logger.info(f"Next run ID is {new_id}")
    return new_id


def get_latest_run_id() -> int:
    data = read_run_info()
    return data.get("last_run_id", 0)


# ------------------- Metadata Helpers -------------------

def save_run_metadata(run_id: int, metadata: dict):
    data = read_run_info()
    if "runs" not in data:
        data["runs"] = {}
    data["runs"][str(run_id)] = metadata
    data["last_run_id"] = max(run_id, data.get("last_run_id", 0))
    write_run_info(data)


def load_run_metadata(run_id: int) -> dict:
    data = read_run_info()
    return data.get("runs", {}).get(str(run_id), {})


def add_step_to_metadata(metadata: dict, step_name: str):
    if "steps" not in metadata:
        metadata["steps"] = []
    metadata["steps"].append({"name": step_name, "time": datetime.now().isoformat()})


def add_artifact_to_metadata(metadata: dict, artifact: Artifact):
    if "artifacts" not in metadata:
        metadata["artifacts"] = {}
    metadata["artifacts"][artifact.name] = str(artifact.get_path())


# ------------------- Utility -------------------

def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_latest_run_artifact(name: str) -> Path:
    """
    Fetches the latest run's artifact by name.
    """
    latest_id = get_latest_run_id()
    if latest_id == 0:
        raise FileNotFoundError("No runs found in run_info.json")
    artifact = Artifact(name=name, type="other", run_id=latest_id)
    return artifact.get_path()



# ------------------- Data Cleaning Functions -------------------

def expand_contractions(text: str) -> str:
    return contractions.fix(text)

def remove_punctuation(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def clean_text(text: str) -> str:
    text = text.lower()
    text = expand_contractions(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)
    return text
