#!/usr/bin/env python3
"""
General inference script - supports custom input and output paths
"""
import argparse
import os
import sys
import subprocess
import time
from pathlib import Path
from threading import Thread
from threading import Lock
import queue

def find_csv_directory(input_dir):
    """Find the directory that contains CSV files"""
    possible_dirs = [
        os.path.join(input_dir, 'csv'),      # original layout
        input_dir,                           # CSVs directly in the directory
        os.path.join(input_dir, 'csvs'),     # csvs directory
        os.path.join(input_dir, 'tables'),   # tables directory
        os.path.join(input_dir, 'data'),     # data directory
    ]
    
    for csv_dir in possible_dirs:
        if os.path.exists(csv_dir) and os.path.isdir(csv_dir):
            csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
            if csv_files:
                return csv_dir, csv_files
    
    return None, []

def load_file_filter(filter_file):
    """Load a filename allowlist used to filter which CSVs are processed"""
    if not filter_file or not os.path.exists(filter_file):
        return None
    
    filter_set = set()
    with open(filter_file, 'r', encoding='utf-8') as f:
        for line in f:
            filename = line.strip()
            if filename:
                # Ensure filenames have a .csv extension
                if not filename.endswith('.csv'):
                    filename += '.csv'
                filter_set.add(filename)
    
    return filter_set

def count_csv_files(input_dir, filter_file=None):
    """Count CSV files for progress display"""
    csv_dir, csv_files = find_csv_directory(input_dir)
    if not csv_files:
        return 0
    
    # If a filter file is provided, count only the filtered files
    if filter_file:
        filter_set = load_file_filter(filter_file)
        if filter_set:
            filtered_files = [f for f in csv_files if f in filter_set]
            return len(filtered_files)
    
    return len(csv_files)

def show_progress_bar(process, total_files, output_file_path):
    """Display a live progress bar based on lines written to the output JSONL file."""
    start_time = time.time()
    last_size = 0
    processed_lines = 0
    bar_length = 40
    spinner = ['|', '/', '-', '\\']
    spin_idx = 0
    print(f"\nStarting with {total_files} tables...")
    while process.poll() is None:
        time.sleep(0.5)
        try:
            # Count new lines appended to the output file since last check
            if os.path.exists(output_file_path):
                current_size = os.path.getsize(output_file_path)
                # Read only the appended part to count newlines
                with open(output_file_path, 'rb') as rf:
                    if last_size > 0:
                        rf.seek(last_size)
                    chunk = rf.read()
                    processed_lines += chunk.count(b"\n")
                    last_size = current_size
        except Exception:
            # Ignore transient read errors while file is being written
            pass
        processed = max(0, min(processed_lines, total_files)) if total_files else processed_lines
        ratio = (processed / total_files) if total_files else 0.0
        filled = int(bar_length * ratio)
        elapsed = time.time() - start_time
        speed = (processed / elapsed) if elapsed > 0 else 0.0
        remaining = max(0.0, (total_files - processed) / speed) if speed > 0 and total_files else 0.0
        bar = '=' * filled + '.' * (bar_length - filled)
        spinner_char = spinner[spin_idx % len(spinner)]
        spin_idx += 1
        line = f"\r[{bar}] {processed}/{total_files} {ratio*100:5.1f}% {spinner_char} | {speed:6.2f}/s | ETA {remaining:6.1f}s"
        print(line, end='', flush=True)
    # Finalize
    process.wait()
    # One last refresh to full bar
    final_bar = '=' * bar_length
    print(f"\r[{final_bar}] {total_files}/{total_files} 100.0% | Done!{' ' * 20}")

def find_repo_root():
    """Automatically find repository root directory"""
    # Check environment variable
    if 'TAB2KNOW_REPO' in os.environ:
        repo_dir = Path(os.environ['TAB2KNOW_REPO'])
        if (repo_dir / 'run_inference.py').exists() and (repo_dir / 'tab2know').exists():
            return repo_dir
    
    # Find from current script location
    script_path = Path(__file__).resolve()
    if script_path.name == 'run_inference.py':
        script_dir = script_path.parent
        if (script_dir / 'tab2know').exists():
            return script_dir
    
    # Walk up from current working directory
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / 'run_inference.py').exists() and (parent / 'tab2know').exists():
            return parent
    
    return None

def main():
    # First find repository root directory
    repo_root = find_repo_root()
    if repo_root is None:
        print("Warning: Cannot automatically find repo root directory, using current directory", file=sys.stderr)
        repo_root = Path.cwd()
    else:
        # Change to repo root directory to ensure relative paths are correct
        os.chdir(repo_root)
    
    parser = argparse.ArgumentParser(description="Tab2Know inference runner")
    parser.add_argument("input_dir", help="Input directory containing tables")
    parser.add_argument("--output", "-o", required=True, help="Output preds.jsonl path")
    parser.add_argument("--modeldir", default="models", help="Model directory (default: models, relative to repo root)")
    parser.add_argument("--type-model", default="supervised-lr", help="Table type model (default: supervised-lr)")
    parser.add_argument("--column-model", default="supervised-lr", help="Column type model (default: supervised-lr)")
    parser.add_argument("--no-caption", action="store_true", help="Ignore table caption (sets TAB2KNOW_NO_CAPTION=1)")
    parser.add_argument("--pythonpath", help="PYTHONPATH to set (default: repo root)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode, disable progress bar")
    parser.add_argument("--filter", "-f", help="Optional filename allowlist (txt, one per line)")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--stall-timeout", type=int, default=120, help="No-progress warning timeout in seconds (default: 900)")
    parser.add_argument("--per-file-timeout", type=int, default=300, help="Per-CSV timeout in seconds; skip on timeout (default: 300)")
    parser.add_argument("--stuck-log", default="results/stuck_tables.txt", help="Path to log stuck tables (default: results/stuck_tables.txt)")
    parser.add_argument("--per-file", action="store_true", help="Enable robust per-file mode with timeouts (slower)")
    parser.add_argument("--max-rows", type=int, default=1000, help="Skip CSVs with more than this number of rows (default: 1000)")
    parser.add_argument("--max-cols", type=int, default=500, help="Skip CSVs with more than this number of columns (default: 500)")
    
    args = parser.parse_args()
    
    # Environment variables
    if args.no_caption:
        os.environ['TAB2KNOW_NO_CAPTION'] = '1'
    
    # Set PYTHONPATH to repo root directory
    pythonpath = args.pythonpath or str(repo_root)
    os.environ['PYTHONPATH'] = pythonpath
    
    # Ensure modeldir is absolute path (relative to repo root)
    if not os.path.isabs(args.modeldir):
        args.modeldir = str(repo_root / args.modeldir)
    
    # Ensure the output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Count CSV files
    total_files = count_csv_files(args.input_dir, args.filter)
    if total_files == 0:
        print("Error: No CSV files found")
        sys.exit(1)
    
    # Resolve file list and optionally build a single filtered temp directory
    temp_dir = None
    csv_dir, csv_files = find_csv_directory(args.input_dir)
    if not csv_dir:
        print("Error: CSV directory not found")
        sys.exit(1)
    filter_set = load_file_filter(args.filter) if args.filter else None
    if filter_set:
        csv_files = [f for f in csv_files if f in filter_set]
        print(f"🔍 Filter mode: selected {len(csv_files)} files")
    input_dir = args.input_dir
    
    def build_cmd(run_input_dir: str) -> str:
        # Use same interpreter as this process (avoid "python: command not found" when only python3 exists)
        modeldir_abs = args.modeldir if os.path.isabs(args.modeldir) else str(repo_root / args.modeldir)
        parts = [
            sys.executable, "-u", str(repo_root / "tab2know" / "main.py"), "run",
            run_input_dir,
            "--modeldir", modeldir_abs,
            "--type-model", args.type_model,
            "--column-model", args.column_model,
            "--output", "metadata",
        ]
        return " ".join(parts)
    print(f"📁 Input dir: {args.input_dir}")
    print(f"📄 Found {len(csv_files)} CSV files")
    
    # Fast size-based prefiltering (columns via first line; rows via chunked newline count with early stop)
    def quick_csv_size_guard(file_path: str, max_rows: int, max_cols: int) -> bool:
        # Return True if file should be kept, False if skipped by limits
        # Check columns using first non-empty line
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore', newline='') as tf:
                for _ in range(100):  # scan up to first 100 lines to find a non-empty header/content
                    line = tf.readline()
                    if not line:
                        break
                    if line.strip():
                        # quick comma-based split; robust enough for a rough guard
                        cols = line.count(',') + 1
                        if cols > max_cols:
                            return False
                        break
        except Exception:
            # If unreadable, keep it to let core pipeline decide
            pass
        # Check rows by counting newlines in chunks with early exit
        if max_rows is not None and max_rows > 0:
            try:
                nl = 0
                with open(file_path, 'rb') as bf:
                    while True:
                        chunk = bf.read(1024 * 1024)
                        if not chunk:
                            break
                        nl += chunk.count(b"\n")
                        if nl > max_rows:
                            return False
            except Exception:
                pass
        return True

    if (args.max_rows and args.max_rows > 0) or (args.max_cols and args.max_cols > 0):
        kept = []
        skipped = 0
        for name in csv_files:
            fp = os.path.join(csv_dir, name)
            if quick_csv_size_guard(fp, args.max_rows, args.max_cols):
                kept.append(name)
            else:
                skipped += 1
        if skipped > 0:
            print(f"⚙️  Size filter active: kept {len(kept)} / {len(csv_files)} (skipped {skipped} by rows>{args.max_rows} or cols>{args.max_cols})")
        csv_files = kept
    print(f"💾 Output file: {args.output}")
    print(f"🧵 Workers: {max(1, args.workers)}")
    
    # Run inference
    try:
        start_time = time.time()
        
        stderr_text = ""
        returncode = 0
        # Ensure the child python process flushes stdout immediately
        child_env = dict(os.environ)
        child_env['PYTHONUNBUFFERED'] = '1'
        
        # Fast default: shard mode unless --per-file is set
        if not args.per_file:
            workers = max(1, int(args.workers))
            base_tmp_root = os.path.join(os.path.dirname(os.path.abspath(args.output)), "temp_shards")
            os.makedirs(base_tmp_root, exist_ok=True)
            # Assign files to shards
            files_per_shard = [list() for _ in range(workers)]
            for idx, name in enumerate(csv_files):
                files_per_shard[idx % workers].append(name)
            shard_outputs = [f"{args.output}.shard{i}" for i in range(workers)]
            for out in shard_outputs:
                open(out, 'w').close()
            # Create shard dirs and launch processes
            processes = []
            for shard_idx in range(workers):
                shard_dir = os.path.join(base_tmp_root, f"shard_{shard_idx}")
                shard_csv_dir = os.path.join(shard_dir, 'csv')
                os.makedirs(shard_csv_dir, exist_ok=True)
                for filename in files_per_shard[shard_idx]:
                    src_path = os.path.join(csv_dir, filename)
                    dst_path = os.path.join(shard_csv_dir, filename)
                    if os.path.exists(dst_path):
                        os.remove(dst_path)
                    try:
                        os.symlink(os.path.abspath(src_path), dst_path)
                    except OSError:
                        import shutil
                        shutil.copy2(src_path, dst_path)
                cmd = build_cmd(shard_dir)
                f = open(shard_outputs[shard_idx], 'w')
                p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.PIPE, text=True, env=child_env, bufsize=1)
                processes.append((p, f))

            # Progress by lines across shard outputs
            start_time = time.time()
            last_total_lines = 0
            last_progress_time = time.time()
            spinner = ['|', '/', '-', '\\']
            spin_idx = 0
            print(f"\nStarting with {len(csv_files)} tables...")
            while True:
                time.sleep(0.5)
                total_lines = 0
                for shard_out in shard_outputs:
                    try:
                        if os.path.exists(shard_out):
                            with open(shard_out, 'rb') as rf:
                                total_lines += rf.read().count(b"\n")
                    except Exception:
                        pass
                processed = max(0, min(total_lines, len(csv_files)))
                ratio = (processed / len(csv_files)) if csv_files else 0.0
                filled = int(40 * ratio)
                elapsed = time.time() - start_time
                speed = (processed / elapsed) if elapsed > 0 else 0.0
                remaining = max(0.0, (len(csv_files) - processed) / speed) if speed > 0 and csv_files else 0.0
                bar = '=' * filled + '.' * (40 - filled)
                spinner_char = spinner[spin_idx % len(spinner)]
                spin_idx += 1
                line = f"\r[{bar}] {processed}/{len(csv_files)} {ratio*100:5.1f}% {spinner_char} | {speed:6.2f}/s | ETA {remaining:6.1f}s"
                print(line, end='', flush=True)
                if processed > last_total_lines:
                    last_total_lines = processed
                    last_progress_time = time.time()
                elif time.time() - last_progress_time > args.stall_timeout:
                    print(f"\n⚠️  No progress for >{args.stall_timeout}s. Check processes or data.")
                    last_progress_time = time.time()
                all_done = all(p.poll() is not None for p, _ in processes)
                if all_done:
                    break

            # Close, collect stderr, and merge outputs
            returncode = 0
            stderr_chunks = []
            for p, f in processes:
                _, shard_err = p.communicate()
                if shard_err:
                    stderr_chunks.append(shard_err)
                if p.returncode != 0:
                    returncode = p.returncode
                f.close()
            stderr_text = "\n".join(stderr_chunks)

            with open(args.output, 'w') as final_out:
                for shard_out in shard_outputs:
                    if os.path.exists(shard_out):
                        with open(shard_out, 'r', encoding='utf-8') as rf:
                            for line in rf:
                                final_out.write(line)

            if returncode != 0:
                print(f"\n❌ Inference failed: {stderr_text.strip()}")
                sys.exit(1)

            end_time = time.time()
            duration = end_time - start_time
            print(f"\n✅ Inference completed!")
            print(f"⏱️  Elapsed: {duration:.1f} s")
            print(f"📊 Avg speed: {total_files/duration:.1f} files/s")
            print(f"💾 Saved to: {args.output}")
            return

        # Switch to per-file worker threads with timeout and stuck logging (slower but robust)
        workers = max(1, int(args.workers))
        shard_outputs = []
        files_per_shard = [list() for _ in range(workers)]
        for idx, name in enumerate(csv_files):
            files_per_shard[idx % workers].append(name)
        base_tmp_root = os.path.join(os.path.dirname(os.path.abspath(args.output)), "temp_shards")
        os.makedirs(base_tmp_root, exist_ok=True)
        for shard_idx in range(workers):
            shard_out = f"{args.output}.shard{shard_idx}"
            open(shard_out, 'w').close()
            shard_outputs.append(shard_out)

        processed_counts = [0 for _ in range(workers)]
        stuck_log_path = getattr(args, 'stuck_log', 'results/stuck_tables.txt')
        Path(stuck_log_path).parent.mkdir(parents=True, exist_ok=True)
        log_lock = Lock()

        def worker(shard_idx: int):
            shard_dir = os.path.join(base_tmp_root, f"shard_{shard_idx}")
            single_dir = os.path.join(shard_dir, 'single')
            single_csv = os.path.join(single_dir, 'csv')
            os.makedirs(single_csv, exist_ok=True)
            out_path = shard_outputs[shard_idx]
            for filename in files_per_shard[shard_idx]:
                # Clean previous file
                for prev in os.listdir(single_csv):
                    try:
                        os.remove(os.path.join(single_csv, prev))
                    except Exception:
                        pass
                src_path = os.path.join(csv_dir, filename)
                dst_path = os.path.join(single_csv, filename)
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except OSError:
                    try:
                        import shutil
                        shutil.copy2(src_path, dst_path)
                    except Exception:
                        pass
                cmd = build_cmd(single_dir)
                try:
                    completed = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=child_env, timeout=getattr(args, 'per_file_timeout', 300))
                    if completed.returncode == 0 and completed.stdout:
                        with open(out_path, 'a', encoding='utf-8') as wf:
                            wf.write(completed.stdout)
                    else:
                        with log_lock:
                            with open(stuck_log_path, 'a', encoding='utf-8') as lf:
                                lf.write(f"timeout_or_error\t{filename}\t{src_path}\tcode={completed.returncode}\n")
                except subprocess.TimeoutExpired:
                    with log_lock:
                        with open(stuck_log_path, 'a', encoding='utf-8') as lf:
                            lf.write(f"timeout\t{filename}\t{src_path}\n")
                finally:
                    processed_counts[shard_idx] += 1

        # Launch threads
        threads = []
        for shard_idx in range(workers):
            t = Thread(target=worker, args=(shard_idx,))
            t.daemon = True
            t.start()
            threads.append(t)

        # Progress aggregation by files processed
        start_time = time.time()
        last_processed = 0
        last_progress_time = time.time()
        spinner = ['|', '/', '-', '\\']
        spin_idx = 0
        total_to_process = len(csv_files)
        print(f"\nStarting with {total_to_process} tables...")
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
            processed = sum(processed_counts)
            ratio = (processed / total_to_process) if total_to_process else 0.0
            filled = int(40 * ratio)
            elapsed = time.time() - start_time
            speed = (processed / elapsed) if elapsed > 0 else 0.0
            remaining = max(0.0, (total_to_process - processed) / speed) if speed > 0 and total_to_process else 0.0
            bar = '=' * filled + '.' * (40 - filled)
            spinner_char = spinner[spin_idx % len(spinner)]
            spin_idx += 1
            line = f"\r[{bar}] {processed}/{total_to_process} {ratio*100:5.1f}% {spinner_char} | {speed:6.2f}/s | ETA {remaining:6.1f}s"
            print(line, end='', flush=True)
            if processed > last_processed:
                last_processed = processed
                last_progress_time = time.time()
            elif time.time() - last_progress_time > args.stall_timeout:
                print(f"\n⚠️  No progress for >{args.stall_timeout}s. Check stuck samples. Log: {stuck_log_path}")
                last_progress_time = time.time()
        # finalize progress
        processed = sum(processed_counts)
        bar = '=' * 40
        print(f"\r[{bar}] {processed}/{total_to_process} 100.0% | Done!{' ' * 20}")

        # Merge shard outputs
        with open(args.output, 'w') as final_out:
            for shard_out in shard_outputs:
                if os.path.exists(shard_out):
                    with open(shard_out, 'r', encoding='utf-8') as rf:
                        for line in rf:
                            final_out.write(line)
        
        if returncode != 0:
            print(f"\n❌ Inference failed: {stderr_text.strip()}")
            sys.exit(1)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Inference completed!")
        print(f"⏱️  Elapsed: {duration:.1f} s")
        print(f"📊 Avg speed: {total_files/duration:.1f} files/s")
        print(f"💾 Saved to: {args.output}")
        
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned temp directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️  Failed to clean temp directory: {e}")
        
    except Exception as e:
        print(f"\n❌ Execution error: {e}")
        # Ensure the temporary directory is cleaned up
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        sys.exit(1)

if __name__ == "__main__":
    main()
