# GitHub Actions Setup Checklist

Complete step-by-step checklist to deploy the K-Number Extractor with GitHub Actions.

## ✅ Phase 1: Repository Setup (5 minutes)

- [ ] **1.1** Initialize git in project (if not done)
  ```bash
  cd /Users/shubham/Downloads/Grassstone/separate/k-number-extractor
  git init
  ```

- [ ] **1.2** Create `.env` from template (LOCAL ONLY - don't commit)
  ```bash
  cp .env.example .env
  # Add your credentials to .env (optional for local testing)
  ```

- [ ] **1.3** Add files to git
  ```bash
  git add .
  git commit -m "Initial commit: K-number extractor with GitHub Actions"
  ```

- [ ] **1.4** Create new repository on GitHub
  - Go to https://github.com/new
  - Name: `k-number-extractor` (or your choice)
  - Description: "Automated FDA 510(k) predicate device extraction pipeline"
  - Visibility: Private (recommended for security)
  - Click "Create repository"

- [ ] **1.5** Add remote and push
  ```bash
  git remote add origin https://github.com/YOUR_USERNAME/k-number-extractor.git
  git branch -M main
  git push -u origin main
  ```

- [ ] **1.6** Verify files in GitHub
  - Go to your repository
  - Verify you can see `k_number_extractor_batch.py`
  - Verify `.github/workflows/` directory exists

---

## ✅ Phase 2: Configure GitHub Secrets (5 minutes)

Go to: https://github.com/YOUR_USERNAME/k-number-extractor/settings/secrets/actions

- [ ] **2.1** Create secret: `ZAI_API_KEY`
  - Click "New repository secret"
  - Name: `ZAI_API_KEY`
  - Value: `98be7dee61b247bdbf7f3210e65c935a.VhXvLLFcYypOcapJ`
  - Click "Add secret"

- [ ] **2.2** Create secret: `SNOWFLAKE_USER`
  - Value: Your Snowflake username
  - Example: `shubham_user`

- [ ] **2.3** Create secret: `SNOWFLAKE_PASSWORD`
  - Value: Your Snowflake password

- [ ] **2.4** Create secret: `SNOWFLAKE_ACCOUNT`
  - Value: Your Snowflake account identifier
  - Format: `xy12345.us-east-1`
  - Get from: Snowflake → Account → Account locator

- [ ] **2.5** Create secret: `SNOWFLAKE_WAREHOUSE`
  - Value: Your warehouse name
  - Example: `COMPUTE_WH`

- [ ] **2.6** Create secret: `SNOWFLAKE_DATABASE`
  - Value: `MEDICAL_DEVICES`

- [ ] **2.7** Create secret: `SNOWFLAKE_SCHEMA`
  - Value: `RAW`

- [ ] **2.8** Verify all secrets are set
  ```bash
  gh secret list
  ```
  Should show 7 secrets

---

## ✅ Phase 3: Verify Workflow (2 minutes)

- [ ] **3.1** Check workflow file exists in repository
  - Go to `.github/workflows/` directory
  - Should see `k-number-extractor-scheduler.yml`

- [ ] **3.2** Enable GitHub Actions (if disabled)
  - Settings → Actions → General
  - "Actions permissions" → "Allow all actions"
  - Click "Save"

- [ ] **3.3** View Actions tab
  - Click "Actions" tab in repository
  - Should see "K-Number Extractor - Scheduled Runner" workflow
  - Status should be "Disabled" or "Enabled"

---

## ✅ Phase 4: Test Workflow (10 minutes)

- [ ] **4.1** Trigger manual test run
  - Go to Actions tab
  - Select "K-Number Extractor - Scheduled Runner"
  - Click "Run workflow" dropdown
  - Click "Run workflow" button
  - Select "main" branch if prompted
  - Click "Run workflow"

- [ ] **4.2** Monitor the run
  - Should see new run appear in list
  - Status: Yellow (running) → Green (success) or Red (failed)
  - Click on run to see details

- [ ] **4.3** Check job logs
  - Click on "extract-predicates" job
  - Expand each step to verify:
    - ✓ Checkout repository
    - ✓ Setup Python
    - ✓ Install dependencies
    - ✓ Configure environment
    - ✓ Run K-Number Extractor
    - ✓ Aggregate results
    - ✓ Commit and push

- [ ] **4.4** Verify results were committed
  ```bash
  git pull origin main
  ls -la results/
  ```
  Should see:
  - `all_extractions.json`
  - `extraction_statistics.json`
  - `daily_extractions/` directory

- [ ] **4.5** Check results content
  ```bash
  cat results/extraction_statistics.json | head -20
  ```
  Should show JSON with statistics

---

## ✅ Phase 5: Enable Automatic Scheduling (2 minutes)

- [ ] **5.1** Verify automatic schedule is enabled
  - Settings → Actions → Runners or Workflow settings
  - Workflow should be enabled (no "Disabled" badge)

- [ ] **5.2** Confirm schedule (every 5 minutes)
  - Open `.github/workflows/k-number-extractor-scheduler.yml`
  - Line: `- cron: '*/5 * * * *'`
  - This runs at: :00, :05, :10, :15, ... each hour

- [ ] **5.3** First automatic run verification
  - Wait 5 minutes for next scheduled run
  - Check Actions tab for new run
  - Verify it completes successfully

---

## ✅ Phase 6: Ongoing Monitoring (ongoing)

Daily Tasks:
- [ ] **6.1** Monitor Actions tab weekly
  - Check success rate
  - Look for any failures
  - Review logs if failures occur

- [ ] **6.2** Review results
  ```bash
  git pull origin main
  cat results/extraction_statistics.json | jq '.'
  ```

- [ ] **6.3** Check storage usage
  ```bash
  du -sh results/
  ```

Monthly Tasks:
- [ ] **6.4** Archive old results (optional)
  - Create backup of `results/daily_extractions/`
  - Remove old dates if storage is concern

- [ ] **6.5** Check GitHub Actions usage
  - Settings → Billing and plans → Usage
  - Verify not exceeding limits

- [ ] **6.6** Rotate API keys
  - Update `ZAI_API_KEY` secret if needed
  - Update `SNOWFLAKE_PASSWORD` periodically

---

## ✅ Phase 7: Optional Optimizations

- [ ] **7.1** Change schedule (if too frequent)
  - Edit `.github/workflows/k-number-extractor-scheduler.yml`
  - Change cron from `*/5 * * * *` to desired schedule
  - Commit and push changes

- [ ] **7.2** Adjust batch size (if timeouts occur)
  - Edit `.github/workflows/k-number-extractor-scheduler.yml`
  - Change `--limit 100` to different number
  - Commit and push changes

- [ ] **7.3** Set up email notifications (GitHub)
  - Settings → Notifications
  - Enable "Workflow runs"

- [ ] **7.4** Create data export script
  - Export `results/all_extractions.json` for analysis
  - Create visualization dashboard

- [ ] **7.5** Set up alerts for failures
  - Settings → Actions → Workflow runs
  - Enable notifications for failures

---

## 🆘 Troubleshooting

### Issue: Workflow doesn't appear in Actions tab

**Solution**:
- [ ] Push the code: `git push`
- [ ] Wait 1-2 minutes
- [ ] Refresh GitHub page
- [ ] Verify `.github/workflows/k-number-extractor-scheduler.yml` exists

### Issue: Secret error in workflow logs

**Solution**:
- [ ] Verify all secrets are set: `gh secret list`
- [ ] Check secret names match exactly (case-sensitive)
- [ ] Recreate secret: `gh secret set SECRET_NAME --body "value"`

### Issue: Snowflake connection failed

**Solution**:
- [ ] Test Snowflake credentials locally
- [ ] Verify account format: `xy12345.us-east-1`
- [ ] Check Snowflake network access (firewalls)
- [ ] Verify warehouse is active

### Issue: API authentication failed

**Solution**:
- [ ] Test API key manually
- [ ] Verify key is not expired
- [ ] Check for typos in API key
- [ ] Request new API key if needed

### Issue: Results not being committed

**Solution**:
- [ ] Check "Commit and push results" step in logs
- [ ] Verify git user is configured in workflow
- [ ] Ensure repository has write permissions
- [ ] Check if there are actual changes to commit

---

## 📋 Quick Reference

### View Latest Statistics
```bash
git pull origin main
cat results/extraction_statistics.json | jq '.'
```

### View All K-numbers Processed
```bash
git pull origin main
cat results/all_extractions.json | jq 'length'
```

### Check Workflow Status
```bash
gh workflow list
gh run list --workflow=k-number-extractor-scheduler.yml --limit 5
```

### Manual Trigger Run
```bash
gh workflow run k-number-extractor-scheduler.yml
```

### View Latest Run Logs
```bash
gh run view --latest --log
```

---

## 📞 Support Resources

- **Main Documentation**: `README.md`
- **Quick Start**: `QUICKSTART.md`
- **GitHub Actions Guide**: `GITHUB_ACTIONS_SETUP.md`
- **Workflow Details**: `.github/WORKFLOW_README.md`
- **Project Structure**: `PROJECT_STRUCTURE.md`

---

## ✨ Success Indicators

Your setup is complete when:

✅ Workflow appears in Actions tab
✅ Manual trigger test run succeeds
✅ Results file (`all_extractions.json`) is created
✅ Statistics file is generated
✅ Automatic runs execute every 5 minutes
✅ Results accumulate over time

---

## 🎉 You're Done!

Your K-Number extraction pipeline is now:
- ✅ Running automatically every 5 minutes
- ✅ Extracting 100 K-numbers per run
- ✅ Accumulating results to master JSON
- ✅ Committing results to your repository
- ✅ Tracking statistics and metrics

**Estimated daily accumulation**: ~28,800 K-numbers processed
**Estimated monthly accumulation**: ~864,000 K-numbers processed

Monitor and enjoy the continuous data collection! 🚀
