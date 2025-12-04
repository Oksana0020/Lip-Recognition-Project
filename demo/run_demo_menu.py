"""Interactive menu to run demo scripts.
Options:
 - initial lip detection
 - extract lip regions with dlib
 - extract words from GRID video
 - extract phonemes with equal-partition method
"""

from pathlib import Path
import subprocess
import sys
import os

DEMOS = [
    ("Initial lip detection (initial_lip_detection.py)", "initial_testing/initial_lip_detection.py", "initial_testing"),
    ("Extract lip regions with dlib (extract_lip_regions_with_dlib.py)", "demo/extract_lip_regions_with_dlib.py", "demo"),
    ("Extract words from GRID video (extract_words_from_grid_video.py)", "demo/extract_words_from_grid_video.py", "demo"),
    ("Extract phonemes (equal partition) (extract_phonemes_equal_partition.py)", "demo/extract_phonemes_equal_partition.py", "demo"),
]

def choose_demo() -> int:
    print("Choose a demo to run:")
    for i, (label, _, _) in enumerate(DEMOS, 1):
        print(f"  {i}. {label}")
    print("  0. Exit")
    while True:
        try:
            choice = int(input("Enter number: ").strip())
        except Exception:
            print("Please enter a number.")
            continue
        if 0 <= choice <= len(DEMOS):
            return choice
        print("Choice out of range.")

def run_demo(script_rel: str, workdir_rel: str):
    repo_root = Path(__file__).parent.parent
    workdir = repo_root / workdir_rel
    script_path = repo_root / script_rel
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return
    if not workdir.exists():
        print(f"Working directory not found: {workdir}, will use repository root instead.")
        workdir = repo_root
    try:
        rel_parts = Path(script_rel).parts
        if len(rel_parts) > 1 and rel_parts[0] == workdir_rel:
            script_arg = rel_parts[-1]
        else:
            script_arg = str(script_path)
    except Exception:
        script_arg = str(script_path)

    cmd = [sys.executable, script_arg]
    env = os.environ.copy()
    env_pythonpath = env.get("PYTHONPATH", "")
    if str(workdir) not in env_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = str(workdir) + (os.pathsep + env_pythonpath if env_pythonpath else "")

    print(f"Running: {' '.join(cmd)} (cwd={workdir})")
    try:
        subprocess.run(cmd, cwd=str(workdir), check=False, env=env)
    except KeyboardInterrupt:
        print("Interrupted by user")

def main():
    while True:
        choice = choose_demo()
        if choice == 0:
            print("Goodbye")
            return
        label, script, workdir = DEMOS[choice - 1]
        print(f"Selected: {label}")
        run_demo(script, workdir)
        print("\nDemo finished.\n")

if __name__ == "__main__":
    main()
