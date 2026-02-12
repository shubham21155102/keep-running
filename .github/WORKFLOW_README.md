# GitHub Actions Workflow

Automated K-Number Predicate Device Extraction Pipeline

## 📋 Overview

- **Schedule**: Runs every 5 minutes automatically
- **Batch Size**: 100 K-numbers per run
- **Results**: Committed to `results/` directory
- **Total Runs/Day**: 288 (24 hours × 12 runs/hour)

## 📁 Files

- **`.github/workflows/k-number-extractor-scheduler.yml`** - Main workflow definition
- **`.github/scripts/aggregate_results.py`** - Results aggregation script
- **`results/all_extractions.json`** - Master results file (auto-updated)
- **`results/extraction_statistics.json`** - Statistics (auto-updated)

## 🔄 Workflow Steps

```
1. Checkout Repository
     ↓
2. Setup Python 3.11
     ↓
3. Install Dependencies (pip install)
     ↓
4. Configure Environment (.env from secrets)
     ↓
5. Run Extractor (python k_number_extractor_batch.py --limit 100)
     ↓
6. Aggregate Results (python .github/scripts/aggregate_results.py)
     ↓
7. Commit & Push to Repository
     ↓
8. Upload Artifacts (for debugging)
```

## ⚙️ Configuration

### Schedule (Cron)

Current: `*/5 * * * *` (every 5 minutes)

Change in `.github/workflows/k-number-extractor-scheduler.yml`:
```yaml
schedule:
  - cron: '*/5 * * * *'  # Edit this
```

### Batch Size

Current: `--limit 100`

Change in workflow file:
```yaml
- name: Run K-Number Extractor
  run: |
    python k_number_extractor_batch.py --limit 100  # Edit this
```

### Timeout

Current: 30 minutes (sufficient for 100 K-numbers)

Change in workflow file:
```yaml
jobs:
  extract-predicates:
    timeout-minutes: 30  # Edit this
```

## 📊 Results

### Master File: `results/all_extractions.json`

Contains all unique K-numbers processed:
```json
[
  {
    "k_number": "K214829",
    "success": true,
    "predicates": ["K190123"],
    "similar_devices": ["K191234"],
    "timestamp": "2024-02-13T10:30:45"
  }
]
```

### Statistics: `results/extraction_statistics.json`

Aggregated metrics:
```json
{
  "total_k_numbers": 28800,
  "successful": 27648,
  "failed": 1152,
  "success_rate": 96.0,
  "unique_predicates": 5000,
  "unique_similar_devices": 3500
}
```

## 🧪 Testing

### Manual Trigger

```bash
# GitHub CLI
gh workflow run k-number-extractor-scheduler.yml

# Or via GitHub UI:
# Actions → K-Number Extractor → Run workflow
```

### View Logs

```bash
# GitHub CLI
gh run list --workflow=k-number-extractor-scheduler.yml
gh run view <RUN_ID> --log

# Or via GitHub UI:
# Actions → K-Number Extractor → Select run → View logs
```

## ✅ Required Setup

Before enabling, ensure:

1. Repository created on GitHub
2. Code pushed to repository
3. All secrets configured (7 total)
4. Workflow file present in `.github/workflows/`
5. Both script files present

### Required Secrets:
- `ZAI_API_KEY`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`

## 📈 Data Accumulation

| Time | K-numbers | Runs | Storage |
|------|-----------|------|---------|
| 1 hour | 600 | 12 | ~300KB |
| 1 day | 28,800 | 288 | ~14MB |
| 1 week | 201,600 | 2,016 | ~100MB |
| 1 month | 864,000 | 8,640 | ~430MB |

## 🔐 Security

- Credentials stored as GitHub Secrets (encrypted)
- Never logged or exposed in workflow output
- Only used to create temporary `.env` file during run
- Results committed with GitHub Actions account

## ⚠️ Costs

### GitHub Actions Usage

- **Free tier**: 2,000 minutes/month
- **Each run**: ~5-10 minutes
- **Per month**: 288 × 10 = 2,880 minutes

⚠️ **This workflow may exceed free tier after ~20 days**

**Recommendation**: Monitor Actions usage in repository Settings

```
Settings → Billing and plans → Usage
```

## 🔄 Manual Pause/Resume

```bash
# Disable workflow
gh workflow disable .github/workflows/k-number-extractor-scheduler.yml

# Enable workflow
gh workflow enable .github/workflows/k-number-extractor-scheduler.yml

# List workflows
gh workflow list
```

## 📝 Monitoring

### Check Last Run

```bash
gh run list --workflow=k-number-extractor-scheduler.yml --limit 1
```

### View Latest Statistics

```bash
git pull origin main
cat results/extraction_statistics.json | jq .
```

### Monitor Success Rate

```bash
git log --oneline results/extraction_statistics.json | head -20
```

## 🆘 Troubleshooting

### Workflow Won't Run

1. Check if workflow file exists: `.github/workflows/k-number-extractor-scheduler.yml`
2. Push changes: `git push`
3. Wait 1-2 minutes for GitHub to recognize workflow
4. Refresh Actions tab

### Secrets Not Found

```bash
# Verify secrets are set
gh secret list

# Recreate a secret
gh secret set ZAI_API_KEY --body "value"
```

### Run Failed

1. Click on failed run in Actions tab
2. Expand "Run K-Number Extractor" step
3. Check error message
4. Common issues:
   - Invalid credentials
   - Snowflake not accessible
   - API rate limiting
   - Network timeout

### Results Not Committed

1. Check "Commit and push results" step in logs
2. Verify git configuration in workflow
3. Ensure repository has write permissions
4. Check if there are actual changes to commit

## 📚 Documentation

- **Main Guide**: `../GITHUB_ACTIONS_SETUP.md`
- **Main README**: `../README.md`
- **Quick Start**: `../QUICKSTART.md`

## 🎯 What's Next?

1. ✅ Push code to GitHub
2. ✅ Configure secrets
3. ✅ Test with manual trigger
4. ✅ Monitor first few runs
5. ✅ Adjust schedule/batch size as needed
6. ✅ Set up notifications (optional)

---

**Workflow Created**: February 13, 2024
**Schedule**: Every 5 minutes
**Batch Size**: 100 K-numbers per run
