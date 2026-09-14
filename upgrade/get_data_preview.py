import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict
import os
import multiprocessing as mp
from tqdm import tqdm

def build_docker_command(compt: str, gpu: int):
    """Build Docker command with only existing directory mounts
    # Run ./zip.sh first (preparation step)
    # Allow docker to accessand mount
    chmod -R a+rw {home_dir}/.cache
    chmod -R a+rw {home_dir}/.cache/mle-bench/data
    # Allow to save files in docker
    chmod -R a+rw {home_dir}/mle-bench-internal/docker-test/scripts
    # Create new docker image with pre pip install via {home_dir}/mle-bench-internal/docker-test/Dockerfile.base
    """

    dst = os.path.abspath(f"../.cache/mle-bench/data/{compt}")

    optional_paths = {
        f"{dst}/prepared/public/train": "/kaggle/input/train/train",
        f"{dst}/prepared/public/train2": "/kaggle/input/train/train2",
        f"{dst}/prepared/public/train2": "/kaggle/input/train2/train2",
        f"{dst}/prepared/public/train_images": "/kaggle/input/train",
        f"{dst}/prepared/public/train_images": "/kaggle/input/train/train",
        f"{dst}/prepared/public/train_images": "/kaggle/input/train/train_images",
        f"{dst}/prepared/public/train_images": "/kaggle/input/train_images/train_images",
        f"{dst}/prepared/public/test_images": "/kaggle/input/test",
        f"{dst}/prepared/public/test_images": "/kaggle/input/test/test",
        f"{dst}/prepared/public/test_images": "/kaggle/input/test/test_images",
        f"{dst}/prepared/public/test_images": "/kaggle/input/test_images/test_images",
        f"{dst}/prepared/public/test": "/kaggle/input/test/test",
        f"{dst}/prepared/public/test2": "/kaggle/input/test/test2",
        f"{dst}/prepared/public/test2": "/kaggle/input/test2/test2"
    }

    volume_mounts =[]

    # Add only existing directories
    for host_path, container_path in optional_paths.items():
        if os.path.exists(host_path) and os.path.isdir(host_path):
            volume_mounts.append(f'"{host_path}:{container_path}"')



    cmd = f'docker run --rm -i --privileged --shm-size=30g'
    cmd += f' --cpuset-cpus="{gpu}"'
    cmd += f' -v {dst}/prepared/public:/kaggle/input'
    cmd += f" -v {dst}/prepared/public:/kaggle/input/{compt}"
    cmd += f' -v {dst}/prepared/public:/kaggle/working/{compt}'
    cmd += f' -v {dst}/prepared/public:/kaggle/data'
    cmd += f' -v {dst}/prepared/public:/kaggle/data/{compt}'
    cmd += "".join([f" -v {mount}" for mount in volume_mounts])
    
    return cmd

def generate_data_previews_for_compt(compt: str, gpu: int) -> Dict[str, str]:
    """
    For a given competition:
    1. Build and run the Docker container
    2. Generate data preview inside container
    3. Save preview as JSON locally
    
    Args:
        compt: Competition name (key from kernel.json)
        output_json_path: Path to save the output JSON file
    
    Returns:
        Dictionary mapping compt -> data_preview string
    """
    
    # print(f"Processing competition: {compt}")
    
    # Build Docker command
    cmd = build_docker_command(compt, gpu)
    
    # Create a temporary script to run inside the container
    preview_script = r"""
import json
from pathlib import Path

import humanize
import pandas as pd
from pandas.api.types import is_numeric_dtype

try:
    from genson import SchemaBuilder
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "genson"])
    from genson import SchemaBuilder

# Only allow traversing under /kaggle, even if we "start at /"
ALLOWED_ROOT = Path("/kaggle")

SKIP_FILES = {"preview_script.py", ".git", "__pycache__", ".ipynb_checkpoints"}

code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files

IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff",
}

# Performance guards
SKIP_DIR_NAMES = {
    "train_images", "test_images",
    "images",
    "train_tfrecords", "test_tfrecords",
}
WALK_MAX_DEPTH = 7               # don't recurse deeply
WALK_MAX_FILES = 10           # hard cap total files visited
WALK_ALLOWED_SUFFIXES = plaintext_files  # only consider these

def _mostly_images_dir(
    dir_path: Path,
    *,
    sample_limit: int = 400,
    min_files: int = 50,
    img_ratio: float = 0.95,
    max_non_img_to_list: int = 20,
):
    total = 0
    img = 0
    img_examples = []
    non_img_examples = []
    dir_examples = []

    try:
        it = dir_path.iterdir()
    except Exception:
        return (False, [], [], [], 0, 0)

    for p in it:
        try:
            if p.is_symlink() or p.name in SKIP_FILES:
                continue
        except Exception:
            continue

        # Collect subfolders (do NOT count towards image ratio sampling)
        try:
            if p.is_dir():
                if len(dir_examples) < max_dirs_to_list:
                    dir_examples.append(p)
                continue
        except Exception:
            continue

        # Sample files for image ratio detection
        total += 1
        is_img = p.suffix.lower() in IMAGE_SUFFIXES
        if is_img:
            img += 1
            if len(img_examples) < 2:
                img_examples.append(p)
        else:
            if len(non_img_examples) < max_non_img_to_list:
                non_img_examples.append(p)

        if total >= sample_limit:
            break

    if total < min_files:
        return (False, img_examples, non_img_examples, dir_examples, total, img)

    return ((img / total) >= img_ratio, img_examples, non_img_examples, dir_examples, total, img)


def get_file_len_size(f: Path) -> tuple[int, str]:
    try:
        if f.suffix in plaintext_files:
            try:
                num_lines = sum(1 for _ in open(f, "rb"))
                return num_lines, f"{num_lines} lines"
            except OSError:
                return 0, "unreadable"
        else:
            try:
                s = f.stat().st_size
                return s, humanize.naturalsize(s)
            except OSError:
                return 0, "unstat-able"
    except Exception:
        return 0, "unknown"


def file_tree(path: Path, depth=0, max_depth=7) -> str:
    if depth > max_depth:
        return f"{' '*depth*4}... (max depth reached)"

    if Path(path) == Path("/"):
        if ALLOWED_ROOT.exists():
            subtree = file_tree(ALLOWED_ROOT, depth=2, max_depth=max_depth)
            lines = ["/", "    kaggle/"]
            if subtree:
                lines.append(subtree)
            return "\n".join(lines)
        return "/"

    # If folder is mostly images: skip ONLY images, but still show non-image files (fast).
def file_tree(path: Path, depth=0, max_depth=6) -> str:
    if depth > max_depth:
        return f"{' '*depth*4}... (max depth reached)"

    if Path(path) == Path("/"):
        if ALLOWED_ROOT.exists():
            subtree = file_tree(ALLOWED_ROOT, depth=2, max_depth=max_depth)
            lines = ["/", "    kaggle/"]
            if subtree:
                lines.append(subtree)
            return "\n".join(lines)
        return "/"

    # If folder is mostly images: skip ONLY images, but still show non-image files (fast).
    if depth > 1:
        # Fast heuristic check first
        is_img_dir, _, _, _, _, _ = _mostly_images_dir(Path(path))
        
        if is_img_dir:
            img_files = []
            non_img_files = []
            subdirs = []
            total_images_count = 0
            
            try:
                # Full pass to separate content correctly
                for p in Path(path).iterdir():
                    if p.is_symlink() or p.name in SKIP_FILES:
                        continue
                        
                    if p.is_dir():
                        subdirs.append(p)
                        continue
                        
                    if p.suffix.lower() in IMAGE_SUFFIXES:
                        total_images_count += 1
                        if len(img_files) < 2:
                            img_files.append(p)
                    else:
                        non_img_files.append(p)
            except Exception:
                return f"{' '*depth*4}... (error reading directory)"

            indent = " " * depth * 4
            lines = []

            # 1. Example images
            for ex in img_files:
                lines.append(f"{indent}{ex.name} ({get_file_len_size(ex)[1]})")

            # 2. Summary of skipped images
            remaining = max(0, total_images_count - len(img_files))
            if remaining > 0:
                lines.append(f"{indent}... and {remaining} other files")

            # 3. Non-image files
            for p in sorted(non_img_files, key=lambda x: x.name):
                lines.append(f"{indent}{p.name} ({get_file_len_size(p)[1]})")

            # 4. Subdirectories
            dirs_to_show = sorted(subdirs, key=lambda x: x.name)
            omitted_dirs = 0
            if len(dirs_to_show) > 10:
                omitted_dirs = len(dirs_to_show) - 2
                dirs_to_show = dirs_to_show[:2]
                
            for d in dirs_to_show:
                lines.append(f"{indent}{d.name}/")
                subtree = file_tree(d, depth + 1, max_depth=max_depth)
                if subtree:
                    lines.append(subtree)
            
            if omitted_dirs > 0:
                lines.append(f"{indent}... and {omitted_dirs} other folders")

            return "\n".join(lines)

    try:
        entries = sorted(Path(path).iterdir())
    except Exception:
        return ""

    result = []

    files = [p for p in entries if not p.is_dir()]
    dirs = [p for p in entries if p.is_dir()]

    # Show all files up through 3rd level; deeper => show up to 2.
    max_n = None if depth <= 3 else 2

    shown = 0
    for p in files:
        if p.name in SKIP_FILES or p.is_symlink():
            continue
        if max_n is not None and shown >= max_n:
            continue
        result_line = f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})"
        shown += 1
        result.append(result_line)

    if max_n is not None:
        eligible = [f for f in files if f.name not in SKIP_FILES]
        if len(eligible) > max_n:
            result.append(f"{' '*depth*4}... and {len(eligible)-max_n} other files")

    # If a folder contains >10 subfolders, only list 2 and summarize the rest.
    dirs_to_show = dirs
    omitted_dirs = 0
    if len(dirs) > 10:
        dirs_to_show = dirs[:2]
        omitted_dirs = len(dirs) - len(dirs_to_show)

    for p in dirs_to_show:
        if p.is_symlink():
            continue
        result.append(f"{' '*depth*4}{p.name}/")
        subtree = file_tree(p, depth + 1, max_depth=max_depth)
        if subtree:
            result.append(subtree)

    if omitted_dirs > 0:
        result.append(f"{' '*depth*4}... and {omitted_dirs} other folders")

    return "\n".join(result)


def _walk(path: Path, depth=0):
    # If called with /, only walk /kaggle
    if Path(path) == Path("/"):
        path = ALLOWED_ROOT

    # stop recursion
    if depth > WALK_MAX_DEPTH:
        return

    try:
        entries = sorted(Path(path).iterdir())
    except Exception:
        return

    for p in entries:
        if p.is_symlink():
            continue
        if p.is_dir():
            if p.name in SKIP_DIR_NAMES and depth >= 3:
                continue
            yield from _walk(p, depth + 1)
            continue

        if p.name in SKIP_FILES:
            continue
        if p.suffix not in WALK_ALLOWED_SUFFIXES:
            continue

        yield p


def preview_csv(p: Path, file_name: str, simple=True) -> str:
    # Fast CSV preview without reading full file into pandas
    try:
        with open(p, "rb") as f:
            # count lines cheaply
            n_lines = sum(1 for _ in f)
        n_rows = max(0, n_lines - 1)
    except OSError:
        n_rows = -1

    try:
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        n_cols = len(cols)
    except Exception:
        cols = []
        n_cols = -1

    out = []
    out.append(f"-> {file_name} has {n_rows} rows and {n_cols} columns.")

    if cols:
        if simple:
            sel_cols = 15
            cols_str = ", ".join(cols[:sel_cols])
            res = f"The columns are: {cols_str}"
            if len(cols) > sel_cols:
                res += f"... and {len(cols)-sel_cols} more columns"
            out.append(res)
        else:
            # small sample for stats
            df = pd.read_csv(p, nrows=2000)
            out.append("Here is some information about the columns:")
            for col in sorted(df.columns):
                dtype = df[col].dtype
                name = f"{col} ({dtype})"
                nan_count = df[col].isnull().sum()

                if dtype == "bool":
                    v = df[col][df[col].notnull()].mean()
                    out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
                elif df[col].nunique() < 10:
                    out.append(f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}")
                elif is_numeric_dtype(df[col]):
                    out.append(f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values")
                elif dtype == "object":
                    out.append(f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}")

    return "\n".join(out)


def preview_json(p: Path, file_name: str):
    # If genson isn't available, just show top-level keys/sample
    try:
        with open(p, "r", encoding="utf-8") as f:
            first = f.readline().strip()
    except OSError:
        return f"-> {file_name} (unreadable JSON)"

    if SchemaBuilder is None:
        try:
            obj = json.loads(first)
            if isinstance(obj, dict):
                keys = list(obj.keys())
                return f"-> {file_name} JSON keys: {keys[:50]}" + (f" ... (+{len(keys)-50} more)" if len(keys) > 50 else "")
        except Exception:
            pass
        return f"-> {file_name} (JSON schema skipped: genson not installed)"

    builder = SchemaBuilder()
    with open(p) as f:
        first_line = first
        try:
            first_object = json.loads(first_line)
            if not isinstance(first_object, dict):
                raise json.JSONDecodeError("The first line isn't JSON", first_line, 0)

            second_line = f.readline().strip()
            if second_line:
                f.seek(0)
                for i, line in enumerate(f):
                    if i > 2000:
                        break
                    builder.add_object(json.loads(line.strip()))
            else:
                builder.add_object(first_object)
        except json.JSONDecodeError:
            f.seek(0)
            builder.add_object(json.load(f))

    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(indent=2)


def generate(base_path, include_file_details=True, simple=False):
    tree = f"```\n{file_tree(base_path)}\n```"
    out = [tree]

    display_root = ALLOWED_ROOT if Path(base_path) == Path("/") else Path(base_path)

    if include_file_details:
        n = 0
        for fn in _walk(base_path, depth=0):
            n += 1
            if n > WALK_MAX_FILES:
                out.append(f"-> (stopped after {WALK_MAX_FILES} files for performance)")
                break

            try:
                file_name = str(fn.relative_to(display_root))
            except Exception:
                file_name = str(fn)

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix in plaintext_files:
                # only inline tiny files
                if get_file_len_size(fn)[0] < 30:
                    try:
                        with open(fn, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                    except OSError:
                        continue
                    if fn.suffix in code_files:
                        content = f"```\n{content}\n```"
                    out.append(f"-> {file_name} has content:\n\n{content}")

    result = "\n\n".join(out)

    if len(result) > 4_000 and not simple:
        return generate(base_path, include_file_details=include_file_details, simple=True)

    return result

data_path = Path("/")
preview = generate(data_path, include_file_details=True, simple=False)
""" + f"""
result = {{"{compt}": preview}}
print(json.dumps(result))
"""
    # Write temp script and copy into container
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(preview_script)
        temp_script = f.name

    docker_cmd = (
        f"{cmd}"
        f" -v {temp_script}:/kaggle/working/preview_script.py:ro"
        f" -w /kaggle/working kaggle_coding"
        f" python /kaggle/working/preview_script.py"
    )
    
    try:
        result = subprocess.run(
            docker_cmd,
            shell=True,
            capture_output=True,
            text=True,
            # timeout=1800,  # was 600
        )
        
        if result.returncode != 0:
            print(f"Error processing {compt}:")
            print(f"stderr: {result.stderr}")
            return {compt: f"Error: {result.stderr}"}
        
        raw = result.stdout.strip()
        json_line = raw.splitlines()[-1]
        # Parse output
        try:
            output = json.loads(json_line)
            # print(json_line)
            return output
        except json.JSONDecodeError:
            print(f"Failed to parse JSON output for {compt}")
            return {compt: result.stdout}
    
    except subprocess.TimeoutExpired:
        return {compt: "Error: Container execution timeout"}
    except Exception as e:
        return {compt: f"Error: {str(e)}"}
    finally:
        Path(temp_script).unlink(missing_ok=True)


def _worker(cpu_id: int, job_q: "mp.Queue", result_q: "mp.Queue") -> None:
    # Best-effort CPU pinning for the Python worker
    try:
        os.sched_setaffinity(0, {cpu_id})
    except Exception:
        pass

    while True:
        compt = job_q.get()
        if compt is None:
            return

        try:
            preview_map = generate_data_previews_for_compt(compt, gpu=cpu_id)  # gpu arg is used as --cpuset-cpus
            preview_text = preview_map.get(compt, "")
            result_q.put((compt, preview_text))
        except Exception as e:
            result_q.put((compt, f"Error: {e}"))


def generate_all_previews(kernel_json_path: Path, output_json_path: Path, output_txt_dir: Path) -> None:
    """
    Load kernel.json, generate previews for all competitions, and save:
      - one human-readable txt per compt in output_txt_dir
      - one aggregated json in output_json_path (incremental)
    """
    output_txt_dir.mkdir(parents=True, exist_ok=True)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    with open(kernel_json_path, "r", encoding="utf-8") as f:
        kernel_meta = json.load(f)

    compts = list(kernel_meta.keys())
    print(f"Found {len(compts)} competitions to process")

    all_previews: Dict[str, str] = {}

    # Distribute onto CPU 0-47 (up to 48 concurrent workers)
    max_workers = 48
    num_workers = min(max_workers, len(compts)) if compts else 0
    cpu_ids = list(range(num_workers))  # 0..num_workers-1 (subset of 0..47)

    job_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()

    existing = {}
    if output_json_path.exists():
        with open(output_json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    all_compts = list(kernel_meta.keys())
    pending_compts = [
        compt
        for compt in all_compts
        if compt not in existing
    ]

    for compt in pending_compts:
        job_q.put(compt)
    
    print("existing:", len(existing))
    print("all compts:", len(all_compts))
    print("pending:", len(pending_compts))
    print("job_q:", job_q.qsize())

    for _ in range(num_workers):
        job_q.put(None)  # sentinel per worker

    procs: list[mp.Process] = []
    print(cpu_ids)
    for cpu_id in cpu_ids:
        p = mp.Process(target=_worker, args=(cpu_id, job_q, result_q), daemon=True)
        p.start()
        procs.append(p)

    # Collect results and write outputs from the parent process (avoids write races)
    for _ in tqdm(range(len(pending_compts)), total=len(pending_compts), desc="Generating previews"):
        compt, preview_text = result_q.get()

        all_previews[compt] = preview_text

        txt_path = output_txt_dir / f"{compt}.txt"
        txt_path.write_text(preview_text, encoding="utf-8")

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(all_previews, f, indent=2, ensure_ascii=False)

    for p in procs:
        p.join()

    # for compt in tqdm(compts, total=len(compts), desc="Generating previews"):
    #     if compt == "nomad2018-predict-transparent-conductors":
    #         preview_map = generate_data_previews_for_compt(compt, gpu=0)  # gpu arg is used as --cpuset-cpus
    #         preview_text = preview_map.get(compt, "")

    #         print(preview_map)
            
    #         print(f"Writing preview for competition: {compt}")
    #         txt_path = output_txt_dir / f"{compt}.txt"
    #         txt_path.write_text(preview_text, encoding="utf-8")
    #         break

    print(f"\nCompleted! TXT previews saved to: {output_txt_dir}")
    print(f"Aggregated JSON saved to: {output_json_path}")


if __name__ == "__main__":
    kernel_json_path = Path("./kernel.json")

    output_json_path = Path("./upgrade/data_preview.json")
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    output_txt_dir = Path("./upgrade/data_previews")
    output_txt_dir.mkdir(parents=True, exist_ok=True)

    generate_all_previews(kernel_json_path, output_json_path, output_txt_dir)