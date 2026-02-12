# K-Number Predicate Device Extractor (Batch Mode)

A production-ready Python tool to extract predicate devices from FDA 510(k) documents using Snowflake and the Z.ai API.

## Features

- **Batch Processing**: Fetch K-numbers from Snowflake and process them automatically
- **Z.ai API Integration**: Uses the Z.ai Claude API for intelligent extraction
- **PDF Processing**: Automatically downloads and extracts text from FDA 510(k) PDFs
- **Vector Search**: Uses FAISS + HuggingFace embeddings for context retrieval
- **Semantic Re-ranking**: Cross-encoder reranking for better result quality
- **Comprehensive Logging**: Detailed progress tracking and error handling
- **JSON Output**: Structured results for downstream processing

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements_extractor.txt
```

### 2. Configure Environment Variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your actual credentials:

```env
# Z.ai API Configuration
ZAI_API_KEY=your_zai_api_key_here

# Snowflake Configuration
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_ACCOUNT=xy12345.us-east-1
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=MEDICAL_DEVICES
SNOWFLAKE_SCHEMA=RAW
```

**Important**: Add `.env` to your `.gitignore` to avoid committing credentials.

## Usage

### Basic Usage (Process all K-numbers from Snowflake)

```bash
python k_number_extractor_batch.py
```

### Process Limited Number of K-numbers

```bash
python k_number_extractor_batch.py --limit 10
```

### Process Specific K-numbers

```bash
python k_number_extractor_batch.py --k-numbers K214829,K221234,K230456
```

### View Help

```bash
python k_number_extractor_batch.py --help
```

## Output Format

The script generates a JSON file with the following structure:

```json
[
  {
    "k_number": "K214829",
    "success": true,
    "predicates": ["K190123", "K180456"],
    "similar_devices": ["K191234", "K201567"],
    "timestamp": "2024-02-13T10:30:45.123456"
  },
  {
    "k_number": "K221234",
    "success": false,
    "error": "PDF not found for K-number",
    "timestamp": "2024-02-13T10:31:20.654321"
  }
]
```

### Output Fields

- **k_number**: The FDA K-number being processed
- **success**: Whether extraction was successful
- **predicates**: List of predicate device K-numbers found
- **similar_devices**: List of similar/reference device K-numbers found
- **error**: Error message if extraction failed
- **timestamp**: ISO format timestamp of processing

## Processing Steps

For each K-number, the extractor performs these steps:

1. **PDF Extraction**: Downloads and extracts text from the FDA 510(k) PDF
2. **Document Chunking**: Splits text into overlapping chunks for better context
3. **Vector Embedding**: Creates embeddings using HuggingFace BGE model
4. **Context Retrieval**: Uses MMR search to find relevant chunks
5. **Re-ranking**: Cross-encoder reranks results for quality
6. **LLM Extraction**: Z.ai API extracts predicate devices using Claude
7. **JSON Parsing**: Parses and validates JSON response

## Performance Notes

- **First Run**: Models are downloaded and cached (~2GB for embeddings and reranker)
- **Processing Speed**: ~2-5 minutes per K-number (depends on PDF size and API latency)
- **GPU Support**: Automatically uses CUDA if available (much faster)
- **Memory**: Requires ~8GB RAM (less with GPU)

## Error Handling

The script handles various error scenarios:

- Missing PDFs (404 errors)
- Corrupt PDF files
- API timeouts
- Invalid K-number formats
- JSON parsing errors

All errors are logged and included in the results file for analysis.

## API Rate Limiting

Be aware of Z.ai API rate limits. For large batches:

```bash
# Process in smaller batches
python k_number_extractor_batch.py --limit 50
# Wait a bit, then process next batch
```

## Troubleshooting

### Snowflake Connection Failed

**Error**: "Missing Snowflake configuration"

**Solution**: Verify all SNOWFLAKE_* environment variables are set in `.env`

```bash
# Test connection
snowsql -c your_connection_name
```

### Out of Memory

**Error**: "CUDA out of memory" or memory errors

**Solutions**:
- Use CPU instead: `export CUDA_VISIBLE_DEVICES=""`
- Reduce batch size: `--limit 5`
- Increase system swap space

### API Authentication Failed

**Error**: "Authentication parameter not received"

**Solution**: Check ZAI_API_KEY is correct and valid

```bash
# Test API manually
curl -X POST "https://api.z.ai/api/anthropic/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_KEY" \
  -d '{"model":"claude-opus-4-5-20251101","max_tokens":100,"messages":[{"role":"user","content":"test"}]}'
```

### PDF Extraction Fails

**Error**: "PDF not found" or "Invalid PDF"

**Possible Causes**:
- K-number doesn't exist in FDA database
- FDA website is down
- K-number format is invalid

Check K-number format: Should be K followed by 6 digits (e.g., K214829)

## Development

### Adding Custom Prompts

Edit the `llm_prompt_content` in the `call_zai_api()` function to customize extraction logic.

### Using Different Models

Modify the API call in `call_zai_api()`:

```python
"model": "claude-opus-4-6",  # Use different Claude version
```

### Extending Results

Add additional fields to the results dictionary in `extract_predicates()`:

```python
results.append({
    'k_number': kNumber,
    'success': True,
    'predicates': result.predicate_devices,
    'similar_devices': result.similar_devices,
    'device_name': device_name,  # Add custom field
    'timestamp': datetime.now().isoformat()
})
```

## Example Workflow

```bash
# 1. Setup
pip install -r requirements_extractor.txt
cp .env.example .env
# Edit .env with credentials

# 2. Test with small batch
python k_number_extractor_batch.py --limit 5

# 3. Check results
cat predicate_extraction_results_*.json | python -m json.tool

# 4. Process full dataset
python k_number_extractor_batch.py

# 5. Analyze results
python -c "
import json
with open('predicate_extraction_results_*.json') as f:
    data = json.load(f)
    successful = sum(1 for r in data if r['success'])
    print(f'Success rate: {successful}/{len(data)} ({successful/len(data)*100:.1f}%)')
"
```

## Security

⚠️ **Important**:

1. **Never commit `.env` files** with credentials
2. **Use strong passwords** for Snowflake
3. **Rotate API keys regularly**
4. **Restrict file permissions**: `chmod 600 .env`
5. **Review output files** before sharing (may contain sensitive data)

## License

This tool is for authorized use only. Ensure compliance with FDA regulations and your organization's policies.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error logs in output JSON
3. Test individual components manually
4. Check Z.ai API status
