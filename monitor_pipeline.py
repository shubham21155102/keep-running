#!/usr/bin/env python3
"""
Monitor GitHub Actions Pipeline for keep-running repository
Continuously checks and displays workflow run status
"""

import subprocess
import json
import time
from datetime import datetime
from typing import List, Dict

REPO = "shubham21155102/keep-running"
CHECK_INTERVAL = 30  # seconds


def get_workflow_runs(limit: int = 15) -> List[Dict]:
    """Fetch recent workflow runs using gh CLI."""
    cmd = [
        "gh", "run", "list",
        "--repo", REPO,
        "--limit", str(limit),
        "--json", "status,conclusion,createdAt,name,workflowName,displayTitle,url,event"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching runs: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []


def format_timestamp(iso_timestamp: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return iso_timestamp[:19]


def get_status_icon(status: str, conclusion: str) -> str:
    """Get status icon."""
    if status == "in_progress":
        return "🔄"
    elif conclusion == "success":
        return "✅"
    elif conclusion == "failure":
        return "❌"
    elif conclusion == "cancelled":
        return "⚠️"
    else:
        return "⏳"


def display_runs(runs: List[Dict]) -> None:
    """Display workflow runs in formatted table."""
    print("\n" + "=" * 80)
    print("  GitHub Actions Pipeline Monitor")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    if not runs:
        print("No workflow runs found.")
        return

    print(f"{'Status':<12} {'Conclusion':<12} {'Time':<20} {'Workflow'}")
    print("-" * 80)

    for run in runs:
        status = run.get('status', 'unknown')
        conclusion = run.get('conclusion', 'N/A')
        workflow = run.get('workflowName', 'Unknown')
        timestamp = format_timestamp(run.get('createdAt', ''))
        url = run.get('url', '')

        icon = get_status_icon(status, conclusion)
        status_formatted = f"{icon} {status.upper()}"
        conclusion_formatted = conclusion.upper() if conclusion != 'N/A' else 'N/A'

        print(f"{status_formatted:<12} {conclusion_formatted:<12} {timestamp:<20} {workflow}")

    print("-" * 80)
    print(f"Total: {len(runs)} runs")
    print()


def print_latest_summary(runs: List[Dict]) -> None:
    """Print summary of latest runs."""
    if not runs:
        return

    completed = [r for r in runs if r.get('status') == 'completed']
    in_progress = [r for r in runs if r.get('status') == 'in_progress']

    success_count = len([r for r in completed if r.get('conclusion') == 'success'])
    failure_count = len([r for r in completed if r.get('conclusion') == 'failure'])

    print(f"📊 Summary:")
    print(f"  ✅ Success: {success_count}")
    print(f"  ❌ Failed: {failure_count}")
    print(f"  🔄 In Progress: {len(in_progress)}")
    print()


def main():
    """Main monitoring loop."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        GitHub Actions Pipeline Monitor - keep-running                       ║
║                                                                      ║
║  Press Ctrl+C to stop monitoring                                        ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    try:
        while True:
            runs = get_workflow_runs(limit=15)
            display_runs(runs)
            print_latest_summary(runs)

            print(f"Next check in {CHECK_INTERVAL} seconds... (Press Ctrl+C to stop)")
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
