# GitHub Actions Setup Guide

Complete guide to set up automated K-Number extraction pipeline in GitHub.

## 📋 Prerequisites

1. **GitHub Repository** - Push this project to GitHub
2. **GitHub Secrets** - Configure required credentials
3. **GitHub Actions** - Enabled in repository (default)

## 🚀 Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit: K-number extractor with GitHub Actions"

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/k-number-extractor.git
git branch -M main
git push -u origin main
```

## 🔐 Step 2: Configure GitHub Secrets

### Via GitHub Web UI:

1. Go to your repository on GitHub
2. **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret below:

### Required Secrets:

#### `ZAI_API_KEY`
```
Value: 98be7dee61b247bdbf7f3210e65c935a.VhXvLLFcYypOcapJ
```

#### `SNOWFLAKE_USER`
```
Value: your_snowflake_username
```

#### `SNOWFLAKE_PASSWORD`
```
Value: your_snowflake_password
```

#### `SNOWFLAKE_ACCOUNT`
```
Value: xy12345.us-east-1
Example: xy12345.us-east-1 (get from Snowflake)
```

#### `SNOWFLAKE_WAREHOUSE`
```
Value: COMPUTE_WH
(or your warehouse name)
```

#### `SNOWFLAKE_DATABASE`
```
Value: MEDICAL_DEVICES
```

#### `SNOWFLAKE_SCHEMA`
```
Value: RAW
```

### Via GitHub CLI:

```bash
gh secret set ZAI_API_KEY --body "98be7dee61b247bdbf7f3210e65c935a.VhXvLLFcYypOcapJ"
gh secret set SNOWFLAKE_USER --body "your_username"
gh secret set SNOWFLAKE_PASSWORD --body "your_password"
gh secret set SNOWFLAKE_ACCOUNT --body "xy12345.us-east-1"
gh secret set SNOWFLAKE_WAREHOUSE --body "COMPUTE_WH"
gh secret set SNOWFLAKE_DATABASE --body "MEDICAL_DEVICES"
gh secret set SNOWFLAKE_SCHEMA --body "RAW"
```

## ✅ Step 3: Verify Secrets

```bash
# List all secrets (doesn't show values for security)
gh secret list
```

You should see:
```
ZAI_API_KEY
SNOWFLAKE_ACCOUNT
SNOWFLAKE_DATABASE
SNOWFLAKE_PASSWORD
SNOWFLAKE_SCHEMA
SNOWFLAKE_USER
SNOWFLAKE_WAREHOUSE
```

## 🔄 Step 4: Enable GitHub Actions

1. Go to **Actions** tab in repository
2. You should see **K-Number Extractor - Scheduled Runner** workflow
3. Click on it and verify it's enabled

## 📅 Step 5: Verify Scheduling

The workflow is configured to run:
- **Automatically**: Every 5 minutes (cron: `*/5 * * * *`)
- **Manually**: Via workflow dispatch button

## 🧪 Step 6: Test the Workflow

### Option A: Manual Trigger
1. Go to **Actions** tab
2. Select **K-Number Extractor - Scheduled Runner**
3. Click **Run workflow** → **Run workflow**
4. Watch the run progress

### Option B: Wait for Next Schedule
The workflow automatically runs at:
- 00:05, 00:10, 00:15, ... (every 5 minutes UTC)

## 📊 Step 7: Monitor Results

### View Run History
1. **Actions** tab → **K-Number Extractor - Scheduled Runner**
2. See all runs with status (✓ success or ✗ failure)

### View Results
Results are stored in the repository:
- **`results/all_extractions.json`** - Master file with all predicates
- **`results/extraction_statistics.json`** - Statistics
- **`results/daily_extractions/YYYY-MM-DD/`** - Daily backups

### Access Results
```bash
# Clone and view results
git clone https://github.com/YOUR_USERNAME/k-number-extractor.git
cat results/all_extractions.json

# View statistics
cat results/extraction_statistics.json
```

## 🔍 Troubleshooting

### Workflow Not Running

**Problem**: Workflow doesn't appear in Actions tab

**Solutions**:
1. Ensure `.github/workflows/k-number-extractor-scheduler.yml` exists
2. Push the workflow file to repository: `git push`
3. Go to Actions tab - should appear in 1-2 minutes

### Authentication Failures

**Error**: "Authentication parameter not received"

**Solutions**:
1. Verify `ZAI_API_KEY` secret is set correctly
2. Double-check API key is not expired
3. Manually test API key locally first

**Error**: "Missing Snowflake configuration"

**Solutions**:
1. Verify all `SNOWFLAKE_*` secrets are set
2. Check Snowflake account format (should be like `xy12345.us-east-1`)
3. Test Snowflake connection locally first

### Memory or Timeout Issues

**Error**: "CUDA out of memory" or timeout after 30 minutes

**Solutions**:
1. The workflow timeout is set to 30 minutes (sufficient for 100 K-numbers)
2. If timeouts occur, reduce batch size in workflow file:
   ```yaml
   - name: Run K-Number Extractor
     run: |
       python k_number_extractor_batch.py --limit 50
   ```

### Results Not Committing

**Problem**: Results file exists but not pushed to repository

**Solutions**:
1. Verify git is configured in workflow (check logs)
2. Add a GitHub token if needed (usually auto-provided)
3. Check repository settings for push permissions

## 📝 Configuration Options

### Change Schedule

Edit `.github/workflows/k-number-extractor-scheduler.yml`:

```yaml
schedule:
  - cron: '*/5 * * * *'  # Every 5 minutes
```

#### Common Cron Examples:

```yaml
# Every 5 minutes
- cron: '*/5 * * * *'

# Every 15 minutes
- cron: '*/15 * * * *'

# Every hour
- cron: '0 * * * *'

# Every 6 hours
- cron: '0 */6 * * *'

# Once a day at 00:00 UTC
- cron: '0 0 * * *'

# Weekdays at 09:00 UTC
- cron: '0 9 * * 1-5'
```

### Change Batch Size

Edit `.github/workflows/k-number-extractor-scheduler.yml`:

```yaml
- name: Run K-Number Extractor
  run: |
    python k_number_extractor_batch.py --limit 100  # Change 100 to desired limit
```

### Enable/Disable Workflow

```bash
# Disable
gh workflow disable k-number-extractor-scheduler.yml

# Enable
gh workflow enable k-number-extractor-scheduler.yml

# List all workflows
gh workflow list
```

## 📊 Understanding Results

### `all_extractions.json`

Master file updated after each run:
```json
[
  {
    "k_number": "K214829",
    "success": true,
    "predicates": ["K190123", "K180456"],
    "similar_devices": ["K191234"],
    "timestamp": "2024-02-13T10:30:45.123456"
  },
  ...
]
```

### `extraction_statistics.json`

Updated after each run:
```json
{
  "timestamp": "2024-02-13T10:35:00",
  "total_k_numbers": 2500,
  "successful": 2400,
  "failed": 100,
  "success_rate": 96.0,
  "new_in_current_run": 100,
  "updated_in_current_run": 5,
  "unique_predicates": 1250,
  "unique_similar_devices": 800,
  "run_info": {
    "github_run_id": "12345678",
    "github_run_number": "42"
  }
}
```

## 📈 Estimated Data Growth

With 5-minute intervals processing 100 K-numbers per run:

- **Per day**: 288 runs × 100 K-numbers = 28,800 K-numbers
- **Per week**: ~201,600 K-numbers
- **Per month**: ~864,000 K-numbers

**Storage**: Each extraction ≈ 200-500 bytes JSON = ~50-100MB per 100k extractions

## 🔒 Security Best Practices

1. **Rotate Secrets Regularly**
   ```bash
   gh secret set ZAI_API_KEY --body "new_api_key"
   ```

2. **Use Repository-Level Secrets** (done above)

3. **Limit Workflow Permissions** (optional)
   - Settings → Actions → Workflow permissions
   - Select "Read and write permissions"

4. **Audit Workflow Runs**
   - Regularly check Actions logs for failures

## 🆘 Getting Help

### View Detailed Logs

1. Go to **Actions** tab
2. Click on failed workflow run
3. Click on **extract-predicates** job
4. Expand any step to see detailed output

### Common Log Locations

- **Dependency installation**: "Install dependencies" step
- **API errors**: "Run K-Number Extractor" step
- **Commit errors**: "Commit and push results" step

## 🎯 Next Steps

1. ✅ Push to GitHub
2. ✅ Configure all secrets
3. ✅ Trigger manual run
4. ✅ Monitor first run
5. ✅ Verify results in repository
6. ✅ Set up automated schedule
7. ✅ Monitor regular runs

## 📞 Support

For issues:
1. Check GitHub Actions logs
2. Review TROUBLESHOOTING section above
3. Test locally first: `python k_number_extractor_batch.py --limit 5`
4. Check repository wiki for updates

---

**Setup Complete!** Your pipeline will now run every 5 minutes, extracting predicate devices and updating the results file.
