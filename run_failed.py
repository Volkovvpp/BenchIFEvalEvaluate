import sys
import json
import csv
from collections import defaultdict
import subprocess

def main():
    errors_csv = "compare_process/elen_errors.csv"

    failed_samples = defaultdict(list)
    try:
        with open(errors_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = row.get("task")
                doc_id = row.get("doc_id")
                if task and doc_id and doc_id.isdigit():
                    failed_samples[task].append(int(doc_id))
    except FileNotFoundError:
        print(f"Error: {errors_csv} not found.")
        sys.exit(1)

    if not failed_samples:
        print("No failed samples found.")
        sys.exit(0)

    # We will pass --samples JSON via CLI to eval.py
    # Since eval.py checks args.start_index > 0, we can just bypass its limitations by calling lm_eval natively
    # Or, we can modify eval.py to accept --samples from passthrough.
    # Actually, eval.py passes `sys.argv.extend(passthrough)`.
    # So if we pass `--samples '{"ifeval": [1, 2]}'` to `eval.py`, it will be in `passthrough` and go to `lm_eval`.
    # Let's run Python eval.py with --samples

    tasks = ",".join(failed_samples.keys())
    samples_json = json.dumps(dict(failed_samples))

    cmd = [
        sys.executable, "eval.py",
        "--tasks", tasks,
        "--start_index", "0",
        "--limit", "1", # Will be ignored or overridden by samples in passthrough or we can just omit limit if we are using samples
        "--samples", samples_json
    ]

    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
