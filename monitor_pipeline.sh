#!/bin/bash
# Monitor GitHub Actions Pipeline for keep-running repository

REPO="shubham21155102/keep-running"
INTERVAL=30  # Check every 30 seconds

echo "=========================================="
echo "  GitHub Actions Pipeline Monitor"
echo "  Repository: $REPO"
echo "  Check interval: ${INTERVAL}s"
echo "=========================================="
echo ""

last_count=0

while true; do
    clear
    echo "=========================================="
    echo "  GitHub Actions Pipeline Monitor"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # Get recent workflow runs
    runs=$(gh run list --repo "$REPO" --limit 15 --json status,conclusion,createdAt,name,workflowName,displayTitle,url)

    # Parse and display
    echo "Recent Workflow Runs:"
    echo "----------------------------------------"

    echo "$runs" | jq -r '.[] |
        "\(.status[0:1]|ascii_upcase)\(.status[1:]) | \(.conclusion // "N/A") | \(.workflowName) | \(.createdAt[0:19])"' |
    while IFS='|' read -r status conclusion workflow timestamp; do
        status=$(echo "$status" | xargs)
        conclusion=$(echo "$conclusion" | xargs)
        workflow=$(echo "$workflow" | xargs)
        timestamp=$(echo "$timestamp" | xargs)

        # Color coding
        if [ "$status" = "IN_PROGRESS" ]; then
            status_icon="🔄"
        elif [ "$conclusion" = "SUCCESS" ]; then
            status_icon="✅"
        elif [ "$conclusion" = "FAILURE" ]; then
            status_icon="❌"
        elif [ "$conclusion" = "CANCELLED" ]; then
            status_icon="⚠️"
        else
            status_icon="⏳"
        fi

        printf "%s %-8s | %-8s | %s\n" "$status_icon" "$status" "$conclusion" "$timestamp"
        echo "  $workflow"
    done

    echo ""
    echo "----------------------------------------"
    echo "Press Ctrl+C to stop monitoring"
    echo "Next check in ${INTERVAL}s..."

    sleep $INTERVAL
done
