# Extraction Results

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
