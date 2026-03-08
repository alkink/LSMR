#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash remote_apply_py_factory_patch.sh /path/to/LSMR
# If path is omitted, current directory is used.

ROOT_DIR="${1:-.}"
TARGET="${ROOT_DIR%/}/nnet/py_factory.py"

if [ ! -f "$TARGET" ]; then
    echo "HATA: hedef dosya bulunamadı: $TARGET"
    exit 1
fi

python3 - "$TARGET" <<'PY'
from pathlib import Path
import re
import sys

target = Path(sys.argv[1])
orig = target.read_text(encoding="utf-8")
text = orig

if "import glob" not in text:
    text = text.replace("import os\n", "import os\nimport glob\n", 1)

if "[NetworkFactory] exact checkpoint bulunamadı" not in text:
    pattern = re.compile(
        r"(?ms)^    def load_params\(self, iteration, is_bbox_only=False\):\n.*?^    def save_params"
    )
    replacement = '''    def load_params(self, iteration, is_bbox_only=False):
        cache_file = system_configs.snapshot_file.format(iteration)

        if not os.path.exists(cache_file):
            snapshot_dir = system_configs.snapshot_dir
            snapshot_name = system_configs.snapshot_name
            # Gizli karakter/sonek farklarında tolerans: <name>_<iter>*.pkl*
            pattern = os.path.join(snapshot_dir, f"{snapshot_name}_{int(iteration)}*.pkl*")
            candidates = sorted(glob.glob(pattern))

            if candidates:
                cache_file = candidates[0]
                print(
                    "[NetworkFactory] exact checkpoint bulunamadı; eşleşen dosya kullanılıyor: {}"
                    .format(cache_file)
                )
            else:
                nearby = sorted(glob.glob(os.path.join(snapshot_dir, f"{snapshot_name}_*.pkl*")))
                nearby_tail = nearby[-8:]
                raise FileNotFoundError(
                    "Checkpoint bulunamadı.\n"
                    "  expected: {}\n"
                    "  absolute: {}\n"
                    "  cwd: {}\n"
                    "  snapshot_dir: {}\n"
                    "  nearby: {}".format(
                        system_configs.snapshot_file.format(iteration),
                        os.path.abspath(system_configs.snapshot_file.format(iteration)),
                        os.getcwd(),
                        snapshot_dir,
                        nearby_tail,
                    )
                )

        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k, v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model.load_state_dict(model_dict)

    def save_params'''
    text, n = pattern.subn(replacement, text, count=1)
    if n != 1:
        raise SystemExit("load_params bloğu bulunamadı; py_factory.py beklenenden farklı.")

if text != orig:
    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(orig, encoding="utf-8")
    target.write_text(text, encoding="utf-8")
    print(f"Patched: {target}")
    print(f"Backup : {backup}")
else:
    print(f"Already patched: {target}")
PY

python3 -m py_compile "$TARGET"
echo "Syntax OK: $TARGET"

