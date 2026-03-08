"""
Prepare blue economy corpus for PelagicGPT fine-tuning.

Extracts text from:
  1. Supabase tables: news_items, signals, decision_lineage, assistant_sessions
  2. Research markdown files from blue-economy-data/research/

Tokenizes with GPT-2 BPE and saves train.bin / val.bin.
"""

import os
import sys
import json
import glob
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv

# Load .env from blue-economy-data project
BLUE_ECON_ROOT = Path('/Users/Bala_1/dev/blue-economy-data')
load_dotenv(BLUE_ECON_ROOT / '.env')

SUPABASE_URL = os.environ.get('VITE_SUPABASE_URL', '')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')

RESEARCH_DIR = BLUE_ECON_ROOT / 'research'
OUTPUT_DIR = Path(__file__).parent
EOT = '<|endoftext|>'


def get_supabase_client():
    """Create Supabase client for data extraction."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("WARNING: Supabase not configured. Skipping DB extraction.")
        return None
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except ImportError:
        print("WARNING: supabase-py not installed. pip install supabase")
        return None


def extract_news(client) -> list[str]:
    """Extract news articles with AI analysis."""
    if not client:
        return []

    documents = []
    try:
        resp = client.table('news_items').select(
            'title, summary, ai_analysis, source, published_at'
        ).order('published_at', desc=True).limit(2000).execute()

        for item in resp.data or []:
            parts = []
            if item.get('title'):
                parts.append(item['title'])
            if item.get('summary'):
                parts.append(item['summary'])
            if item.get('ai_analysis'):
                parts.append(item['ai_analysis'])

            if parts:
                text = f"{EOT}\n[MARKET COMMENTARY]\n" + '\n'.join(parts)
                documents.append(text)

        print(f"  Extracted {len(documents)} news articles")
    except Exception as e:
        print(f"  WARNING: Failed to extract news: {e}")

    return documents


def extract_signals(client) -> list[str]:
    """Extract signal rationales."""
    if not client:
        return []

    documents = []
    try:
        resp = client.table('signals').select(
            'name, direction, confidence, rationale, tradeable_instruments'
        ).execute()

        for item in resp.data or []:
            if not item.get('rationale'):
                continue

            instruments = ', '.join(item.get('tradeable_instruments') or [])
            text = (
                f"{EOT}\n[SIGNAL NARRATIVE]\n"
                f"Signal: {item['name']} | Direction: {item.get('direction', 'neutral')} | "
                f"Strength: {item.get('confidence', 0)}\n"
            )
            if instruments:
                text += f"Instruments: {instruments}\n"
            text += item['rationale']
            documents.append(text)

        print(f"  Extracted {len(documents)} signal narratives")
    except Exception as e:
        print(f"  WARNING: Failed to extract signals: {e}")

    return documents


def extract_decision_lineage(client) -> list[str]:
    """Extract trade idea theses."""
    if not client:
        return []

    documents = []
    try:
        resp = client.table('decision_lineage').select(
            'ticker, direction, thesis_summary, contributing_signals, created_at'
        ).order('created_at', desc=True).limit(500).execute()

        for item in resp.data or []:
            if not item.get('thesis_summary'):
                continue

            signals_str = ', '.join(item.get('contributing_signals') or [])
            text = (
                f"{EOT}\n[PRICE ANALYSIS]\n"
                f"Ticker: {item['ticker']} | Direction: {item.get('direction', 'long')}\n"
            )
            if signals_str:
                text += f"Contributing signals: {signals_str}\n"
            text += item['thesis_summary']
            documents.append(text)

        print(f"  Extracted {len(documents)} trade theses")
    except Exception as e:
        print(f"  WARNING: Failed to extract decision lineage: {e}")

    return documents


def extract_sessions(client) -> list[str]:
    """Extract assistant session messages."""
    if not client:
        return []

    documents = []
    try:
        resp = client.table('assistant_sessions').select(
            'messages, title'
        ).order('updated_at', desc=True).limit(200).execute()

        for item in resp.data or []:
            messages = item.get('messages') or []
            if not messages:
                continue

            # Extract only assistant responses (the quality content)
            assistant_texts = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    if content and not content.startswith('Error:'):
                        assistant_texts.append(content)

            if assistant_texts:
                text = f"{EOT}\n[MARKET COMMENTARY]\n" + '\n\n'.join(assistant_texts)
                documents.append(text)

        print(f"  Extracted {len(documents)} session transcripts")
    except Exception as e:
        print(f"  WARNING: Failed to extract sessions: {e}")

    return documents


def extract_research_docs() -> list[str]:
    """Read research markdown files."""
    documents = []
    md_files = sorted(RESEARCH_DIR.glob('*.md'))

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding='utf-8')
            if len(content) < 100:
                continue

            # Split large research docs into ~2000 char chunks at paragraph boundaries
            paragraphs = content.split('\n\n')
            chunk = []
            chunk_len = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                if chunk_len + len(para) > 2000 and chunk:
                    text = f"{EOT}\n[MARKET COMMENTARY]\n" + '\n\n'.join(chunk)
                    documents.append(text)
                    chunk = []
                    chunk_len = 0

                chunk.append(para)
                chunk_len += len(para)

            if chunk:
                text = f"{EOT}\n[MARKET COMMENTARY]\n" + '\n\n'.join(chunk)
                documents.append(text)

            print(f"  {md_file.name}: {len(content):,} chars")
        except Exception as e:
            print(f"  WARNING: Failed to read {md_file.name}: {e}")

    print(f"  Extracted {len(documents)} research chunks")
    return documents


def main():
    print("=== PelagicGPT Data Preparation ===\n")

    all_documents = []

    # 1. Extract from Supabase
    print("Connecting to Supabase...")
    client = get_supabase_client()

    print("Extracting news articles...")
    all_documents.extend(extract_news(client))

    print("Extracting signal narratives...")
    all_documents.extend(extract_signals(client))

    print("Extracting trade theses...")
    all_documents.extend(extract_decision_lineage(client))

    print("Extracting session transcripts...")
    all_documents.extend(extract_sessions(client))

    # 2. Extract research docs
    print("\nReading research documents...")
    all_documents.extend(extract_research_docs())

    if not all_documents:
        print("\nERROR: No documents extracted. Check Supabase credentials and research directory.")
        sys.exit(1)

    # 3. Combine into single corpus
    corpus = '\n'.join(all_documents) + f'\n{EOT}\n'
    print(f"\nTotal corpus: {len(corpus):,} characters, {len(all_documents)} documents")

    # 4. Tokenize with GPT-2 BPE
    print("Tokenizing with GPT-2 BPE...")
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(corpus, allowed_special={EOT})
    print(f"Total tokens: {len(tokens):,}")

    # 5. Split 90/10 train/val
    n = len(tokens)
    split_idx = int(n * 0.9)
    train_ids = np.array(tokens[:split_idx], dtype=np.uint16)
    val_ids = np.array(tokens[split_idx:], dtype=np.uint16)

    # 6. Save
    train_path = OUTPUT_DIR / 'train.bin'
    val_path = OUTPUT_DIR / 'val.bin'
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"\ntrain.bin: {len(train_ids):,} tokens ({train_path.stat().st_size / 1024:.0f} KB)")
    print(f"val.bin:   {len(val_ids):,} tokens ({val_path.stat().st_size / 1024:.0f} KB)")
    print("\nDone! Ready for training.")


if __name__ == '__main__':
    main()
