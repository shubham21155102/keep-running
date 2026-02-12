# Quick Start Guide

## 📋 Prerequisites

- Python 3.8+
- Snowflake account with MEDICAL_DEVICES.RAW.RAW_510K table access
- Z.ai API key

## ⚡ 5-Minute Setup

### 1. Clone and Setup

```bash
# Navigate to project directory
cd k-number-extractor

# Run setup script
bash setup.sh

# Or use Makefile
make setup
```

### 2. Configure Credentials

Edit `.env` file with your credentials:

```bash
nano .env
```

Required variables:
```env
ZAI_API_KEY=your_zai_api_key
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=xy12345.us-east-1
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=MEDICAL_DEVICES
SNOWFLAKE_SCHEMA=RAW
```

### 3. Test Run

```bash
# Test with 5 K-numbers
python k_number_extractor_batch.py --limit 5

# Or use Makefile
make test
```

### 4. Full Run

```bash
# Process all K-numbers
python k_number_extractor_batch.py

# Or use Makefile
make run
```

## 📁 Project Structure

```
k-number-extractor/
├── k_number_extractor_batch.py    # Main script
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment template
├── .env                          # Your credentials (don't commit!)
├── .gitignore                    # Git ignore rules
├── setup.sh                      # Setup script
├── Makefile                      # Common commands
├── config.yaml.example           # Config template
├── README.md                     # Full documentation
└── QUICKSTART.md                 # This file
```

## 🚀 Common Commands

```bash
# Setup environment
make setup

# Install dependencies
make install

# Run in test mode (limit 5)
make test

# Run with specific limit
make run-limit N=20

# Run full extraction
make run

# Clean temporary files
make clean

# View help
make help
```

## 📊 What You Get

After running, you'll get a JSON file like `predicate_extraction_results_20240213_103045.json`:

```json
[
  {
    "k_number": "K214829",
    "success": true,
    "predicates": ["K190123", "K180456"],
    "similar_devices": ["K191234"],
    "timestamp": "2024-02-13T10:30:45.123456"
  }
]
```

## ⚠️ Troubleshooting

### "Authentication parameter not received"
- Check `ZAI_API_KEY` in `.env`
- Verify API key is valid

### "Missing Snowflake configuration"
- Verify all `SNOWFLAKE_*` variables in `.env`
- Check account format: `xy12345.us-east-1`

### Out of Memory
```bash
# Run with smaller batches
python k_number_extractor_batch.py --limit 5

# Use CPU (slower but uses less RAM)
export CUDA_VISIBLE_DEVICES=""
```

## 📚 Full Documentation

See `README.md` for comprehensive documentation including:
- Detailed setup instructions
- API configuration
- Error handling
- Performance optimization
- Development guide

## 💡 Tips

1. **First run**: Models will download (~2GB). Be patient!
2. **Rate limiting**: Z.ai API may have limits. Process in batches.
3. **GPU**: If you have CUDA, it's much faster. Install `faiss-gpu`:
   ```bash
   pip install faiss-gpu==1.7.4
   ```
4. **Monitor output**: Check the JSON file as processing happens

## 🆘 Need Help?

1. Check `README.md` for detailed troubleshooting
2. Review error messages in output JSON
3. Test API manually:
   ```bash
   curl -X POST "https://api.z.ai/api/anthropic/v1/messages" \
     -H "x-api-key: YOUR_KEY" \
     -H "Content-Type: application/json" \
     -d '{"model":"claude-opus-4-5-20251101","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}'
   ```

---

**Ready to start?** Run `make setup` and follow the prompts!
