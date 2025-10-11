
import os, json, random, datetime, subprocess, sys, pathlib
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def set_seed(seed: int = 42):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def timestamped_dir(base='outputs'):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base, ts)
    ensure_dir(path)
    return path

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def get_versions():
    def ver(modname):
        try:
            mod = __import__(modname)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return "not-installed"
    return {
        "python": sys.version,
        "numpy": ver("numpy"),
        "pandas": ver("pandas"),
        "sklearn": ver("sklearn"),
        "tensorflow": ver("tensorflow"),
        "keras_tuner": ver("keras_tuner"),
    }

def get_git_commit():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None
