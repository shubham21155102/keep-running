# Project Structure

## 📂 Directory Layout

```
k-number-extractor/
│
├── 📄 k_number_extractor_batch.py      # Main application script
│   ├── Snowflake connection
│   ├── PDF extraction logic
│   ├── Vector store creation
│   ├── Z.ai API integration
│   └── Results aggregation
│
├── 📋 Configuration Files
│   ├── .env.example                    # Environment variables template
│   ├── config.yaml.example             # Advanced configuration template
│   ├── requirements.txt                # Python dependencies
│   └── .gitignore                      # Git ignore rules
│
├── 🚀 Setup & Build
│   ├── setup.sh                        # Automated setup script
│   └── Makefile                        # Common command shortcuts
│
├── 📚 Documentation
│   ├── README.md                       # Full documentation
│   ├── QUICKSTART.md                   # Quick start guide
│   └── PROJECT_STRUCTURE.md            # This file
│
└── 📊 Runtime Output (created during execution)
    ├── predicate_extraction_results_*.json
    ├── .env                            # Your actual credentials
    ├── venv/                           # Virtual environment
    ├── logs/                           # Log files
    └── models/                         # Cached ML models
```

## 📋 File Descriptions

### Core Application

**`k_number_extractor_batch.py`** (15.8 KB)
- Main application entry point
- Implements the complete extraction pipeline:
  1. Snowflake K-number fetching
  2. FDA PDF downloading
  3. Document chunking and embedding
  4. Vector store creation (FAISS)
  5. Context retrieval with re-ranking
  6. Z.ai API calls for LLM extraction
  7. Results compilation and JSON export
- Supports command-line arguments for flexibility
- Comprehensive error handling and logging

### Configuration

**`.env.example`**
- Template for environment variables
- Copy to `.env` and fill with your credentials
- **Never commit `.env` to version control**

**`config.yaml.example`**
- Advanced configuration options
- Customize API settings, models, processing parameters
- Optional: Replace hardcoded values in main script

**`requirements.txt`**
- Python package dependencies
- Install with: `pip install -r requirements.txt`
- Includes:
  - Langchain ecosystem
  - FAISS vector store
  - HuggingFace embeddings
  - Snowflake connector
  - PDF processing libraries

**`.gitignore`**
- Prevents committing sensitive files:
  - `.env` (credentials)
  - Output JSON files
  - Virtual environment
  - Cache and temporary files
  - IDE files

### Setup & Automation

**`setup.sh`**
- Automated initial setup script
- Creates Python virtual environment
- Installs dependencies
- Sets up `.env` file from template
- Verifies Python version

**`Makefile`**
- Common command shortcuts
- Available commands:
  ```
  make setup         # Full setup
  make install       # Install dependencies
  make test          # Test run (5 K-numbers)
  make run           # Full run
  make run-limit N=X # Run with limit
  make clean         # Clean temp files
  make venv          # Create virtual env
  make help          # Show all commands
  ```

### Documentation

**`README.md`** (6.8 KB)
- Comprehensive documentation
- Setup instructions
- Usage examples
- Output format specification
- Troubleshooting guide
- API rate limiting info
- Development extensions

**`QUICKSTART.md`** (3.6 KB)
- 5-minute setup guide
- Basic usage examples
- Common commands
- Troubleshooting quick fixes
- Recommended for first-time users

**`PROJECT_STRUCTURE.md`** (This file)
- Directory and file organization
- File descriptions and purposes
- Data flow overview
- Development guide

## 🔄 Data Flow

```
┌─────────────────┐
│   Snowflake     │
│  RAW_510K       │
└────────┬────────┘
         │
         ├─ K-numbers list
         │
┌────────▼────────┐
│  FDA Website    │
│  (PDF Downloads)│
└────────┬────────┘
         │
         ├─ Raw PDF Text
         │
┌────────▼────────┐
│ Text Processing │
│ (Chunking)      │
└────────┬────────┘
         │
         ├─ Document chunks
         │
┌────────▼────────┐
│  Embeddings     │
│  (HuggingFace)  │
└────────┬────────┘
         │
         ├─ Vector embeddings
         │
┌────────▼────────┐
│  FAISS Store    │
│  (Vector DB)    │
└────────┬────────┘
         │
         ├─ Relevant chunks (MMR)
         │
┌────────▼────────┐
│  Re-ranker      │
│  (CrossEncoder) │
└────────┬────────┘
         │
         ├─ Top-k relevant chunks
         │
┌────────▼────────┐
│   Z.ai API      │
│  (Claude LLM)   │
└────────┬────────┘
         │
         ├─ Extracted JSON
         │
┌────────▼────────┐
│   Results File  │
│   (JSON)        │
└─────────────────┘
```

## 🔧 Configuration Priority

1. **Command-line arguments** (highest priority)
   ```bash
   python k_number_extractor_batch.py --limit 10
   ```

2. **Environment variables** (`.env` file)
   ```env
   ZAI_API_KEY=xxx
   SNOWFLAKE_USER=yyy
   ```

3. **Hardcoded defaults** (in script)
   - Model names
   - Chunk sizes
   - Timeout values

## 📦 Dependencies Overview

### Core Libraries
- **langchain** - LLM framework and document processing
- **faiss-cpu** - Vector similarity search
- **sentence-transformers** - Embedding models
- **pdfplumber** - PDF text extraction

### Integration
- **snowflake-connector** - Snowflake database connection
- **requests** - HTTP client for API calls

### Utilities
- **python-dotenv** - Environment variable management
- **pydantic** - Data validation

## 🚀 Getting Started

### One-Command Setup
```bash
bash setup.sh
```

### Manual Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure credentials
cp .env.example .env
nano .env  # Edit with your credentials

# Run test
python k_number_extractor_batch.py --limit 5
```

### Using Makefile
```bash
make setup
make test
```

## 🔐 Security Considerations

1. **Never commit `.env`** - It contains credentials
2. **Use `.gitignore`** - Already configured
3. **Restrict file permissions**: `chmod 600 .env`
4. **Rotate API keys** - Periodically update Z.ai keys
5. **Secure Snowflake credentials** - Use strong passwords

## 📊 Output Structure

Generated JSON file contains:
```json
{
  "k_number": "K######",
  "success": true|false,
  "predicates": ["K######", ...],
  "similar_devices": ["K######", ...],
  "error": "error message (if failed)",
  "timestamp": "ISO-8601 format"
}
```

## 🎯 Typical Workflow

```
1. Clone/Navigate to project
2. Run: bash setup.sh
3. Edit: .env with credentials
4. Test: make test (5 K-numbers)
5. Verify: Check output JSON
6. Run: make run (all K-numbers)
7. Analyze: Review results JSON
8. Export: Use JSON for downstream processing
```

## 📈 Performance Expectations

- **Setup time**: 5-10 minutes (first time, includes model downloads)
- **Per K-number**: 2-5 minutes (depends on PDF size)
- **Memory usage**: 8GB RAM (less with GPU)
- **GPU speedup**: 2-3x faster with CUDA

## 🔄 Maintenance

### Regular Tasks
- Update Snowflake connection credentials
- Rotate Z.ai API keys
- Monitor disk space for results files
- Archive old result files

### Troubleshooting
- Check `.env` file is properly configured
- Verify Snowflake and Z.ai connectivity
- Review error messages in results JSON
- Check system memory and disk space

## 📚 Additional Resources

- Full README: `README.md`
- Quick Start: `QUICKSTART.md`
- Z.ai API: https://api.z.ai
- Langchain Docs: https://python.langchain.com
- Snowflake Python Connector: https://docs.snowflake.com/python

---

**Last Updated**: February 13, 2024
**Project Version**: 1.0.0
