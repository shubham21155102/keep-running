#!/usr/bin/env python3
"""
Aggregate results from extraction runs.
Combines all extraction results into a master JSON file and tracks statistics.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def load_json_file(filepath: str) -> List[Dict]:
    """Load JSON file safely."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return []

def ensure_results_dir() -> Path:
    """Ensure results directory exists."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    return results_dir

def aggregate_results():
    """Aggregate all extraction results into master file."""
    print("[*] Starting results aggregation...")

    results_dir = ensure_results_dir()
    master_file = results_dir / "all_extractions.json"
    stats_file = results_dir / "extraction_statistics.json"

    # Load existing master results
    existing_results = {}
    if master_file.exists():
        data = load_json_file(str(master_file))
        for item in data:
            existing_results[item['k_number']] = item
        print(f"[+] Loaded {len(existing_results)} existing results")

    # Find all extraction result files in current directory
    result_files = list(Path("results").glob("predicate_extraction_results_*.json"))
    print(f"[+] Found {len(result_files)} extraction result file(s)")

    new_count = 0
    updated_count = 0

    # Process each result file
    for result_file in result_files:
        print(f"\n[*] Processing {result_file.name}")
        results = load_json_file(str(result_file))

        for result in results:
            k_number = result.get('k_number')

            if k_number in existing_results:
                # Update if success, keep old if current failed
                if result.get('success'):
                    existing_results[k_number] = result
                    updated_count += 1
                    print(f"  ✓ Updated: {k_number}")
            else:
                # New result
                existing_results[k_number] = result
                if result.get('success'):
                    new_count += 1
                    print(f"  ✓ New: {k_number}")
                else:
                    print(f"  ✗ Failed: {k_number}")

    # Save updated master file
    master_results = list(existing_results.values())
    master_results.sort(key=lambda x: x['k_number'])

    with open(master_file, 'w') as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[+] Saved master file: {master_file}")

    # Generate statistics
    stats = generate_statistics(master_results, new_count, updated_count)

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[+] Saved statistics: {stats_file}")

    # Archive the extraction result files
    archive_results_files(result_files, results_dir)

    print("\n[✓] Aggregation complete!")
    print(f"\nStatistics:")
    print(f"  Total K-numbers processed: {len(master_results)}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  New in this run: {new_count}")
    print(f"  Updated in this run: {updated_count}")

def generate_statistics(results: List[Dict], new_count: int, updated_count: int) -> Dict[str, Any]:
    """Generate extraction statistics."""
    total = len(results)
    successful = sum(1 for r in results if r.get('success'))
    failed = total - successful

    # Aggregate predicates, similar, parent, and child devices
    all_predicates = set()
    all_similar = set()
    all_parent = set()
    all_child = set()

    for result in results:
        if result.get('success'):
            all_predicates.update(result.get('predicates', []))
            all_similar.update(result.get('similar_devices', []))
            all_parent.update(result.get('parent_devices', []))
            all_child.update(result.get('child_devices', []))

    return {
        "timestamp": datetime.now().isoformat(),
        "total_k_numbers": total,
        "successful": successful,
        "failed": failed,
        "success_rate": (successful / total * 100) if total > 0 else 0,
        "new_in_current_run": new_count,
        "updated_in_current_run": updated_count,
        "unique_predicates": len(all_predicates),
        "unique_similar_devices": len(all_similar),
        "unique_parent_devices": len(all_parent),
        "unique_child_devices": len(all_child),
        "run_info": {
            "github_server": os.getenv('GITHUB_SERVER_URL', 'N/A'),
            "github_repository": os.getenv('GITHUB_REPOSITORY', 'N/A'),
            "github_run_id": os.getenv('GITHUB_RUN_ID', 'N/A'),
            "github_run_number": os.getenv('GITHUB_RUN_NUMBER', 'N/A'),
            "github_ref": os.getenv('GITHUB_REF', 'N/A'),
        }
    }

def archive_results_files(result_files: List[Path], results_dir: Path):
    """Archive individual result files by date."""
    if not result_files:
        return

    archive_dir = results_dir / "daily_extractions"
    archive_dir.mkdir(exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = archive_dir / today
    today_dir.mkdir(exist_ok=True)

    for result_file in result_files:
        try:
            # Copy file to archive
            import shutil
            archived_path = today_dir / result_file.name
            shutil.copy2(result_file, archived_path)

            # Remove original
            result_file.unlink()
            print(f"  📦 Archived: {result_file.name}")
        except Exception as e:
            print(f"  ⚠️ Could not archive {result_file.name}: {e}")

def create_readme():
    """Create README for results directory."""
    results_dir = Path("results")
    readme_path = results_dir / "README.md"

    if not readme_path.exists():
        readme_content = """# Extraction Results

This directory contains aggregated results from the K-Number Predicate Device Extractor pipeline.

## Files

- **all_extractions.json** - Master file with all extracted predicate devices
- **extraction_statistics.json** - Aggregated statistics and metrics
- **daily_extractions/** - Daily backup of extraction runs

## Format

Each entry in `all_extractions.json`:
```json
{
  "k_number": "K######",
  "success": true,
  "predicates": ["K######", ...],
  "similar_devices": ["K######", ...],
  "timestamp": "ISO-8601 format"
}
```

## Statistics

See `extraction_statistics.json` for:
- Total K-numbers processed
- Success rate
- Unique predicates found
- Run information

## Updating Frequency

Results are updated every 5 minutes via GitHub Actions scheduled workflow.

Last updated: Check `extraction_statistics.json` timestamp field.
"""
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"\n[+] Created results README")

if __name__ == "__main__":
    try:
        aggregate_results()
        create_readme()
    except Exception as e:
        print(f"\n[✗] Aggregation failed: {e}")
        exit(1)
