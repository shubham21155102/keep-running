#!/usr/bin/env python3
"""
K-Number Predicate Device Extractor (Batch Mode)
Fetches K-numbers from Snowflake and extracts predicate devices using Z.ai API.

Usage:
    python k_number_extractor_batch.py [--limit 10] [--skip-processed]

Environment variables required:
    - SNOWFLAKE_USER
    - SNOWFLAKE_PASSWORD
    - SNOWFLAKE_ACCOUNT
    - SNOWFLAKE_WAREHOUSE
    - SNOWFLAKE_DATABASE
    - SNOWFLAKE_SCHEMA
    - ZAI_API_KEY
"""

import os
import re
import requests
import pdfplumber
import json
import sys
import argparse
import subprocess
from io import BytesIO
from typing import List, Dict, Any, Optional
from datetime import datetime

# Langchain and related imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

# Snowflake imports
import snowflake.connector

# Torch for device detection
import torch

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configuration
TOKENIZERS_PARALLELISM = True
ZAI_API_KEY = os.getenv("ZAI_API_KEY")
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/anthropic")
ZAI_API_URL = f"{ZAI_BASE_URL}/v1/messages"
LLM_MODEL = os.getenv("LLM_MODEL", "claude-opus-4-5-20251101")

if not ZAI_API_KEY:
    raise ValueError("Please set ZAI_API_KEY environment variable.")

# Snowflake configuration
SNOWFLAKE_CONFIG = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA", "RAW"),
}

# Validate Snowflake config
missing_config = [k for k, v in SNOWFLAKE_CONFIG.items() if not v]
if missing_config:
    raise ValueError(f"Missing Snowflake configuration: {', '.join(missing_config)}")

print(f"✓ Z.ai API configured")
print(f"✓ Snowflake configured for {SNOWFLAKE_CONFIG['account']}")

# Determine device for embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device for embeddings: {device}")

# Global models
embeddings = None
reranker = None


def load_models():
    """Load ML models (embeddings and reranker)."""
    global embeddings, reranker

    if embeddings is not None:
        return

    print("\nLoading ML models (this may take a moment on first run)...")
    try:
        model_name = "BAAI/bge-base-en"
        model_kwargs = {'device': device}
        encode_kwargs = {'batch_size': 56, 'device': device, 'normalize_embeddings': True}

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        print("✓ HuggingFaceEmbeddings model loaded successfully.")

        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✓ CrossEncoder reranker loaded successfully.")

    except Exception as e:
        print(f"Error loading models: {e}")
        raise RuntimeError("Failed to load necessary ML models.")


class PredicateDetails(BaseModel):
    predicate_devices: List[str] = Field(
        default_factory=list,
        description="List of predicate device K-numbers."
    )
    similar_devices: List[str] = Field(
        default_factory=list,
        description="List of similar or equivalent device K-numbers."
    )
    parent_devices: List[str] = Field(
        default_factory=list,
        description="List of parent/reference device K-numbers."
    )
    child_devices: List[str] = Field(
        default_factory=list,
        description="List of child device K-numbers."
    )


class EnhancedGDNCRetriever(BaseRetriever):
    base_retriever: BaseRetriever = Field(...)
    k: int = 5
    rerank: bool = True

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        raw_docs = self.base_retriever.invoke(query)
        filtered_docs = raw_docs
        if self.rerank and filtered_docs:
            pairs = [(query, doc.page_content) for doc in filtered_docs]
            try:
                scores = reranker.predict(pairs)
            except Exception as e:
                print(f"Error during reranker prediction: {e}")
                raise RuntimeError("Failed to rerank documents.")
            scored_docs = sorted(zip(scores, filtered_docs),
                                 key=lambda x: x[0], reverse=True)
            filtered_docs = [doc for _, doc in scored_docs][:self.k]

        return filtered_docs[:self.k]


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def get_context(knumber: str, retriever: EnhancedGDNCRetriever) -> str:
    """Fetches and returns context using the provided retriever."""
    prompt = f"""
    Get all the relevant docs with 510(k) numbers, PREDICATE DEVICE INFORMATION, Primary Predicate Device Name,
    Reference Device Name, Primary Predicate Device Name for docs {knumber}.
    """
    pmt = prompt.strip()
    docs_cfr = retriever.get_relevant_documents(pmt)
    return format_docs(docs_cfr)


def get_fda_headers():
    """Return headers for FDA requests."""
    return {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "accept-language": "en-GB,en;q=0.9",
        "cache-control": "max-age=0",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    }


def extract_pdf_content(kNumber: str) -> Optional[str]:
    """Fetches and extracts text content from PDF for the given K-number."""
    if kNumber.upper().startswith("K"):
        year = kNumber[1:3]  # Extract year from kNumber
        pdf_link = f"https://www.accessdata.fda.gov/cdrh_docs/pdf{year}/{kNumber}.pdf"
    else:
        # Handle other cases like PMA numbers
        pdf_link = f"https://www.accessdata.fda.gov/cdrh_docs/pdfma/{kNumber}.pdf"

    try:
        headers = get_fda_headers()
        response = requests.get(pdf_link, headers=headers, stream=True, timeout=60)
        response.raise_for_status()

        pdf_bytes = BytesIO(response.content)
        extracted_text = ""

        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"

        if not extracted_text.strip():
            return None

        return extracted_text

    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"    ✗ PDF not found for K-number {kNumber}")
        else:
            print(f"    ✗ HTTP {response.status_code} fetching PDF")
        return None
    except Exception as e:
        print(f"    ✗ Error processing PDF: {e}")
        return None


def call_zai_api(context: str, kNumber: str) -> Optional[str]:
    """Call Z.ai API to extract predicate devices."""
    llm_prompt_content = f"""
    You are an expert on CDRH 510(k) device predicates and references—especially experienced
    in extracting FDA 510(k) numbers and relationships from regulatory documents.

    From the context below, extract ALL device K-numbers into these 4 categories:

    1. **predicate_devices** — The primary predicate device(s) listed under "PREDICATE DEVICE" or "Primary Predicate Device Name". These are the devices that the subject device claims substantial equivalence to.
    2. **similar_devices** — Any devices listed as "similar", "equivalent", or "substantially equivalent" that are NOT the primary predicate.
    3. **parent_devices** — Any "Reference Device" or parent device K-numbers. These are devices that the predicate itself was cleared against (the predicate's own predicates).
    4. **child_devices** — Any child or downstream device K-numbers that reference the subject device as their predicate.

    Rules:
    - Only include valid K-numbers (format: K followed by 6-7 digits).
    - Do NOT include the subject device {kNumber} itself in any list.
    - If a category has no matches, return an empty list.
    - Return ONLY the JSON, no explanation.

    Context:
    {context}

    Question: Extract all predicate, similar, parent, and child device K-numbers for {kNumber}.

    Answer:
    ```json
    {{
        "predicate_devices": [ "K#######", ... ],
        "similar_devices": [ "K#######", ... ],
        "parent_devices": [ "K#######", ... ],
        "child_devices": [ "K#######", ... ]
    }}
    ```
    """

    headers = {
        "Content-Type": "application/json",
        "x-api-key": ZAI_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": LLM_MODEL,
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": llm_prompt_content}
        ]
    }

    try:
        response = requests.post(ZAI_API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()

        result = response.json()

        if "content" in result and len(result["content"]) > 0:
            return result["content"][0]["text"]
        else:
            print(f"    ✗ Unexpected Z.ai API response format")
            return None

    except Exception as e:
        print(f"    ✗ Error calling Z.ai API: {e}")
        return None


def extract_json_from_response(response_text: str) -> Optional[Dict]:
    """Extract JSON from LLM response."""
    json_match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)

    if json_match:
        json_string = json_match.group(1).strip()
    else:
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            json_string = json_match.group(0)
        else:
            return None

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def extract_predicates(kNumber: str) -> Optional[PredicateDetails]:
    """Main extraction function that processes a K-number and returns predicate devices."""

    # Ensure models are loaded
    load_models()

    print(f"\n{'─'*60}")
    print(f"Processing: {kNumber}")
    print(f"{'─'*60}")

    # Step 1: Extract PDF content
    print("  [1/5] Extracting PDF content...")
    pdf_content = extract_pdf_content(kNumber)

    if not pdf_content:
        print(f"  ✗ Failed to extract PDF content")
        return None

    print(f"  ✓ Extracted {len(pdf_content)} characters")

    # Step 2: Create document chunks
    print("  [2/5] Creating document chunks...")
    documents = [Document(page_content=pdf_content)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=300,
        separators=["\n\n", "\n", " ", ""],
    )
    cfr_texts = text_splitter.split_documents(documents)
    print(f"  ✓ Created {len(cfr_texts)} chunks")

    # Step 3: Create vector store
    print("  [3/5] Creating FAISS vector store...")
    vectordb_cfr = FAISS.from_documents(
        documents=cfr_texts,
        embedding=embeddings
    )
    print(f"  ✓ Vector store created")

    # Step 4: Retrieve relevant context
    print("  [4/5] Retrieving relevant context...")
    retriever_cfr = EnhancedGDNCRetriever(
        base_retriever=vectordb_cfr.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "score_threshold": 0.9, "lambda_mult": 0.7}
        ),
        k=3,
        rerank=True
    )

    context = get_context(kNumber, retriever_cfr)
    print(f"  ✓ Retrieved context ({len(context)} characters)")

    # Step 5: Call Z.ai API
    print("  [5/5] Extracting with Z.ai API...")
    llm_response = call_zai_api(context, kNumber)

    if not llm_response:
        print(f"  ✗ Z.ai API call failed")
        return None

    print(f"  ✓ API response received")

    # Extract JSON from response
    parsed_json = extract_json_from_response(llm_response)

    if not parsed_json:
        print(f"  ✗ Failed to parse JSON from response")
        return None

    result = PredicateDetails(
        predicate_devices=parsed_json.get("predicate_devices", []),
        similar_devices=parsed_json.get("similar_devices", []),
        parent_devices=parsed_json.get("parent_devices", []),
        child_devices=parsed_json.get("child_devices", [])
    )

    print(f"  ✓ Extraction complete")
    print(f"    - Predicates: {len(result.predicate_devices)}")
    print(f"    - Similar devices: {len(result.similar_devices)}")
    print(f"    - Parent devices: {len(result.parent_devices)}")
    print(f"    - Child devices: {len(result.child_devices)}")

    return result


def get_progress_file_path() -> str:
    """Get the path to the progress tracking file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "results", ".processed_progress.json")


def load_progress_data() -> Dict:
    """Load the entire progress data including fetch tracking."""
    progress_file = get_progress_file_path()

    if not os.path.exists(progress_file):
        return {
            "fetch_offset": 0,
            "batch_size": 100,
            "processed_k_numbers": [],
            "total_processed": 0,
            "last_updated": None
        }

    try:
        with open(progress_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  ⚠ Could not load progress file: {e}")
        return {
            "fetch_offset": 0,
            "batch_size": 100,
            "processed_k_numbers": [],
            "total_processed": 0,
            "last_updated": None
        }


def save_progress_data(progress_data: Dict) -> None:
    """Save the entire progress data."""
    progress_file = get_progress_file_path()
    results_dir = os.path.dirname(progress_file)
    os.makedirs(results_dir, exist_ok=True)

    try:
        progress_data["last_updated"] = datetime.now().isoformat()
        progress_data["total_processed"] = len(progress_data.get("processed_k_numbers", []))

        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    except Exception as e:
        print(f"  ⚠ Could not save progress file: {e}")


def load_processed_k_numbers() -> set:
    """Load the set of already processed K-numbers."""
    progress_data = load_progress_data()
    return set(progress_data.get("processed_k_numbers", []))


def save_processed_k_numbers(processed_set: set) -> None:
    """Save the set of processed K-numbers (updates fetch offset)."""
    progress_data = load_progress_data()
    progress_data["processed_k_numbers"] = sorted(list(processed_set))
    save_progress_data(progress_data)


def fetch_k_numbers_from_snowflake(limit: Optional[int] = None, skip_processed: bool = True, batch_size: int = 100) -> List[str]:
    """Fetch K-numbers from Snowflake with pagination."""
    # Load progress to get current offset
    progress_data = load_progress_data()
    offset = progress_data.get("fetch_offset", 0)

    # Use provided limit or batch_size from progress or default
    if limit:
        fetch_limit = limit
    else:
        fetch_limit = progress_data.get("batch_size", batch_size)

    print("\n" + "="*60)
    print("Fetching K-numbers from Snowflake")
    print("="*60)
    print(f"  Offset: {offset}")
    print(f"  Batch size: {fetch_limit}")

    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()

        query = f"""
        SELECT DISTINCT K_NUMBER
        FROM {SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.RAW_510K
        WHERE K_NUMBER IS NOT NULL AND K_NUMBER != ''
            AND UPPER(K_NUMBER) LIKE 'K%'
            AND SUBSTRING(K_NUMBER, 2, 2) >= '20'  -- Extract year: K20xxxx → '20', K99xxxx → '99'
        ORDER BY K_NUMBER ASC
        LIMIT {fetch_limit} OFFSET {offset}
        """

        print(f"  Executing query...")
        cursor.execute(query)

        k_numbers = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        # Filter only K-numbers starting with 'K' (case-insensitive)
        k_numbers = [k for k in k_numbers if k.upper().startswith('K')]

        print(f"  ✓ Fetched {len(k_numbers)} K-numbers from Snowflake (starting with 'K')")

        # Skip already processed K-numbers BEFORE updating offset
        if skip_processed:
            processed_set = load_processed_k_numbers()
            if processed_set:
                original_count = len(k_numbers)
                k_numbers = [k for k in k_numbers if k not in processed_set]
                skipped_count = original_count - len(k_numbers)
                skip_percentage = (skipped_count / original_count * 100) if original_count > 0 else 0
                print(f"  ✓ Skipped {skipped_count} already processed K-numbers ({skip_percentage:.1f}%)")
                print(f"  📊 Total processed: {len(processed_set)}")

                # Only update offset if less than 90% were skipped
                # If >90% are skipped, we're in already-processed territory, don't advance
                if skip_percentage < 90:
                    new_offset = offset + fetch_limit
                    progress_data["fetch_offset"] = new_offset
                    progress_data["batch_size"] = fetch_limit
                    save_progress_data(progress_data)
                    print(f"  ✓ Fetch offset updated to: {new_offset}")
                else:
                    print(f"  ⚠ High skip rate ({skip_percentage:.1f}%), not advancing offset")
                    print(f"  ℹ Reusing current offset: {offset}")
            else:
                # No processed K-numbers yet, update offset normally
                new_offset = offset + fetch_limit
                progress_data["fetch_offset"] = new_offset
                progress_data["batch_size"] = fetch_limit
                save_progress_data(progress_data)
                print(f"  ✓ Fetch offset updated to: {new_offset}")
        else:
            # Not skipping processed, update offset normally
            new_offset = offset + fetch_limit
            progress_data["fetch_offset"] = new_offset
            progress_data["batch_size"] = fetch_limit
            save_progress_data(progress_data)
            print(f"  ✓ Fetch offset updated to: {new_offset}")

        print(f"  📋 To process: {len(k_numbers)} K-numbers")
        print("="*60)

        return k_numbers

    except Exception as e:
        print(f"  ✗ Error connecting to Snowflake: {e}")
        raise


def save_results(results: List[Dict], filename: str = None) -> str:
    """Save results to a JSON file in the results/ directory."""
    # Get the script directory and results folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    if filename is None:
        filename = f"predicate_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {filepath}")
    return filepath


def commit_and_push_results(filepath: str) -> bool:
    """Commit and push the results file to the remote repository."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        print("\n" + "="*60)
        print("Committing and pushing to repository...")
        print("="*60)

        # Check if we're in a git repository
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print("  ✗ Not a git repository, skipping commit/push")
            return False

        # Check if there are changes to commit
        result = subprocess.run(
            ["git", "status", "--porcelain", "results/"],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if not result.stdout.strip():
            print("  ℹ No changes to commit in results/")
            return False

        # Add the results file
        print("  [1/4] Adding results file to git...")
        result = subprocess.run(
            ["git", "add", filepath],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"  ✗ Failed to add file: {result.stderr}")
            return False
        print("  ✓ File added")

        # Create commit message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        commit_msg = f"🔄 Update extracted predicate devices - {timestamp}"

        # Commit the changes
        print("  [2/4] Committing changes...")
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            # Check if nothing to commit
            if "nothing to commit" in result.stdout.lower():
                print("  ℹ No changes to commit")
                return False
            print(f"  ✗ Failed to commit: {result.stderr}")
            return False
        print("  ✓ Changes committed")

        # Push the changes
        print("  [3/4] Pushing to remote...")
        result = subprocess.run(
            ["git", "push"],
            cwd=script_dir,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"  ✗ Failed to push: {result.stderr}")
            print("  ℹ Changes committed locally but not pushed")
            return False
        print("  ✓ Changes pushed to remote")

        print("  [4/4] Done!")
        print("="*60)
        return True

    except Exception as e:
        print(f"  ✗ Error during git operations: {e}")
        return False


def main():
    """Main batch processing function."""
    parser = argparse.ArgumentParser(
        description="Batch K-number predicate device extractor using Z.ai API"
    )
    parser.add_argument("--limit", type=int, default=None, help="Override batch size for this run")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for pagination (default: 100)")
    parser.add_argument("--skip-processed", action="store_true", default=True, help="Skip K-numbers already processed (default: True)")
    parser.add_argument("--include-processed", action="store_true", help="Include already processed K-numbers")
    parser.add_argument("--k-numbers", type=str, help="Comma-separated list of K-numbers to process")
    parser.add_argument("--no-git", action="store_true", help="Skip git commit and push operations")
    parser.add_argument("--reset-pagination", action="store_true", help="Reset fetch offset to 0")

    args = parser.parse_args()

    # Handle pagination reset
    if args.reset_pagination:
        progress_data = load_progress_data()
        progress_data["fetch_offset"] = 0
        save_progress_data(progress_data)
        print("✓ Fetch offset reset to 0")

    print("\n" + "="*60)
    print("  K-NUMBER PREDICATE DEVICE EXTRACTOR (BATCH MODE)")
    print("="*60)

    # Get K-numbers
    if args.k_numbers:
        k_numbers = [k.strip().upper() for k in args.k_numbers.split(",")]
        # Filter to only K-numbers starting with 'K'
        k_numbers = [k for k in k_numbers if k.upper().startswith('K')]
        print(f"\nProcessing {len(k_numbers)} specified K-numbers (starting with 'K')")
    else:
        # Determine skip_processed flag
        skip_processed = not args.include_processed
        batch_size = args.limit if args.limit else args.batch_size
        k_numbers = fetch_k_numbers_from_snowflake(limit=batch_size, skip_processed=skip_processed)

    if not k_numbers:
        print("No K-numbers to process.")
        return

    # Load existing progress
    processed_set = load_processed_k_numbers()
    if processed_set:
        print(f"📊 Progress tracking: {len(processed_set)} K-numbers already processed")

    # Process results
    results = []
    successful = 0
    failed = 0
    newly_processed = set()

    print(f"\n{'='*60}")
    print(f"Processing {len(k_numbers)} K-numbers...")
    print(f"{'='*60}")

    for i, kNumber in enumerate(k_numbers, 1):
        print(f"\n[{i}/{len(k_numbers)}]", end=" ")

        try:
            result = extract_predicates(kNumber)

            if result:
                results.append({
                    'k_number': kNumber,
                    'success': True,
                    'predicates': result.predicate_devices,
                    'similar_devices': result.similar_devices,
                    'parent_devices': result.parent_devices,
                    'child_devices': result.child_devices,
                    'timestamp': datetime.now().isoformat()
                })
                successful += 1
                # Mark as processed
                newly_processed.add(kNumber)
            else:
                results.append({
                    'k_number': kNumber,
                    'success': False,
                    'error': 'Extraction failed',
                    'timestamp': datetime.now().isoformat()
                })
                failed += 1
                # Still mark as processed (attempted)
                newly_processed.add(kNumber)

        except Exception as e:
            print(f"\n✗ Error processing {kNumber}: {e}")
            results.append({
                'k_number': kNumber,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {len(k_numbers)}")
    print(f"Successful: {successful} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Success rate: {successful/len(k_numbers)*100:.1f}%")

    # Update progress tracking
    if newly_processed:
        processed_set.update(newly_processed)
        save_processed_k_numbers(processed_set)
        print(f"\n📊 Progress updated: {len(processed_set)} total K-numbers processed")
        print(f"  New in this run: {len(newly_processed)}")

    # Save results
    if results:
        filepath = save_results(results)
        print(f"\nResults summary:")
        print(f"  Total entries: {len(results)}")
        print(f"  File size: {os.path.getsize(filepath) / 1024:.1f} KB")

        # Commit and push to repository unless --no-git flag is set
        if not args.no_git:
            commit_and_push_results(filepath)
        else:
            print("\nℹ Git operations skipped (--no-git flag set)")


if __name__ == "__main__":
    main()
