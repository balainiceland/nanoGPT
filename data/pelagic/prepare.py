"""
Prepare blue economy corpus for PelagicGPT fine-tuning.

Extracts text from:
  1. Supabase tables: news_items, signals, decision_lineage, assistant_sessions
  2. Research markdown files from blue-economy-data/research/
  3. IOC seafood map: methodology docs, source logs, enriched factory data, citations

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
IOC_ROOT = Path('/Users/Bala_1/dev/ioc-seafood-map')
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
            'signal_name, direction, confidence, strength, rationale, tradeable_instruments'
        ).execute()

        for item in resp.data or []:
            if not item.get('rationale'):
                continue

            instruments = ', '.join(item.get('tradeable_instruments') or [])
            text = (
                f"{EOT}\n[SIGNAL NARRATIVE]\n"
                f"Signal: {item['signal_name']} | Direction: {item.get('direction', 'neutral')} | "
                f"Strength: {item.get('strength') or item.get('confidence', 0)}\n"
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


def extract_vessels(client) -> list[str]:
    """Extract vessel registry and fleet intelligence."""
    if not client:
        return []

    documents = []
    try:
        resp = client.table('vessels').select(
            'name, mmsi, imo, flag_country, vessel_type, length_m, tonnage_gt, '
            'target_species, owner, is_active'
        ).eq('is_active', True).order('name').execute()

        # Group vessels by flag country for fleet summaries
        by_flag: dict[str, list] = {}
        for v in resp.data or []:
            flag = v.get('flag_country') or 'Unknown'
            by_flag.setdefault(flag, []).append(v)

        # Individual vessel profiles
        for v in resp.data or []:
            species = ', '.join(v.get('target_species') or [])
            parts = [f"Vessel: {v.get('name', 'Unknown')}"]
            if v.get('mmsi'):
                parts.append(f"MMSI: {v['mmsi']}")
            if v.get('flag_country'):
                parts.append(f"Flag: {v['flag_country']}")
            if v.get('vessel_type'):
                parts.append(f"Type: {v['vessel_type']}")
            if v.get('length_m'):
                parts.append(f"Length: {v['length_m']}m")
            if v.get('tonnage_gt'):
                parts.append(f"Tonnage: {v['tonnage_gt']} GT")
            if species:
                parts.append(f"Target species: {species}")
            if v.get('owner'):
                parts.append(f"Owner: {v['owner']}")

            text = f"{EOT}\n[VESSEL INTELLIGENCE]\n" + ' | '.join(parts)
            documents.append(text)

        # Fleet summaries by flag
        for flag, vessels in by_flag.items():
            if len(vessels) < 2:
                continue
            types = {}
            for v in vessels:
                vt = v.get('vessel_type') or 'unknown'
                types[vt] = types.get(vt, 0) + 1

            type_str = ', '.join(f"{count} {t}" for t, count in sorted(types.items(), key=lambda x: -x[1]))
            text = (
                f"{EOT}\n[VESSEL INTELLIGENCE]\n"
                f"Fleet summary for {flag}: {len(vessels)} active vessels. "
                f"Composition: {type_str}."
            )
            documents.append(text)

        print(f"  Extracted {len(documents)} vessel records")
    except Exception as e:
        print(f"  WARNING: Failed to extract vessels: {e}")

    return documents


def extract_vessel_events(client) -> list[str]:
    """Extract vessel events (fishing, port visits, encounters, loitering, AIS gaps)."""
    if not client:
        return []

    documents = []
    try:
        resp = client.table('vessel_events').select(
            'event_type, start_time, end_time, duration_hours, lat, lon, '
            'region_name, port_name, port_flag, encountered_vessel_mmsi, '
            'gap_hours, vessel_mmsi'
        ).order('start_time', desc=True).limit(2000).execute()

        EVENT_DESCRIPTIONS = {
            'fishing': 'fishing activity detected',
            'port_visit': 'port visit recorded',
            'encounter': 'at-sea encounter with another vessel',
            'loitering': 'loitering behavior detected',
            'ais_gap': 'AIS transponder gap detected',
        }

        batch = []
        batch_len = 0

        for ev in resp.data or []:
            etype = ev.get('event_type', 'unknown')
            desc = EVENT_DESCRIPTIONS.get(etype, etype)
            parts = [f"Event: {desc}"]

            if ev.get('vessel_mmsi'):
                parts.append(f"Vessel MMSI: {ev['vessel_mmsi']}")
            if ev.get('region_name'):
                parts.append(f"Region: {ev['region_name']}")
            if ev.get('start_time'):
                parts.append(f"Time: {ev['start_time'][:10]}")
            if ev.get('duration_hours'):
                parts.append(f"Duration: {ev['duration_hours']:.1f}h")
            if ev.get('port_name'):
                parts.append(f"Port: {ev['port_name']} ({ev.get('port_flag', '')})")
            if ev.get('encountered_vessel_mmsi'):
                parts.append(f"Encountered vessel: {ev['encountered_vessel_mmsi']}")
            if ev.get('gap_hours'):
                parts.append(f"AIS gap: {ev['gap_hours']:.1f}h")

            entry = ' | '.join(parts)
            if batch_len + len(entry) > 2000 and batch:
                text = f"{EOT}\n[VESSEL INTELLIGENCE]\n" + '\n'.join(batch)
                documents.append(text)
                batch = []
                batch_len = 0
            batch.append(entry)
            batch_len += len(entry)

        if batch:
            text = f"{EOT}\n[VESSEL INTELLIGENCE]\n" + '\n'.join(batch)
            documents.append(text)

        print(f"  Extracted {len(documents)} vessel event chunks")
    except Exception as e:
        print(f"  WARNING: Failed to extract vessel events: {e}")

    return documents


def extract_fishing_effort(client) -> list[str]:
    """Extract aggregated fishing effort data."""
    if not client:
        return []

    documents = []
    try:
        resp = client.table('fishing_effort').select(
            'region_name, date, fishing_hours, total_hours, distance_km, avg_speed_knots'
        ).order('date', desc=True).limit(2000).execute()

        # Group by region for narrative summaries
        by_region: dict[str, list] = {}
        for row in resp.data or []:
            region = row.get('region_name') or 'Unknown'
            by_region.setdefault(region, []).append(row)

        for region, rows in by_region.items():
            total_fishing = sum(r.get('fishing_hours', 0) or 0 for r in rows)
            total_distance = sum(r.get('distance_km', 0) or 0 for r in rows)
            avg_speed = sum(r.get('avg_speed_knots', 0) or 0 for r in rows) / max(len(rows), 1)
            days = len(rows)

            text = (
                f"{EOT}\n[VESSEL INTELLIGENCE]\n"
                f"Fishing effort summary for {region}: "
                f"{total_fishing:.0f} fishing hours over {days} vessel-days, "
                f"{total_distance:.0f} km covered, "
                f"average speed {avg_speed:.1f} knots."
            )
            documents.append(text)

        print(f"  Extracted {len(documents)} fishing effort summaries")
    except Exception as e:
        print(f"  WARNING: Failed to extract fishing effort: {e}")

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


def extract_ioc_docs() -> list[str]:
    """Extract methodology docs, source logs, and enriched data from IOC seafood map."""
    documents = []

    if not IOC_ROOT.exists():
        print("  WARNING: IOC seafood map not found at", IOC_ROOT)
        return documents

    # Directories to skip (not domain-relevant content)
    skip_dirs = {'.venv', 'node_modules', '.git', '__pycache__', 'dist', '.next'}

    def should_skip(path: Path) -> bool:
        return any(part in skip_dirs for part in path.parts)

    # 1. Methodology and source documentation (markdown files)
    md_files = [f for f in list(IOC_ROOT.glob('*.md')) + list(IOC_ROOT.glob('**/*source_log*.md'))
                if not should_skip(f)]
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding='utf-8')
            if len(content) < 100:
                continue

            # Chunk at paragraph boundaries
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

            print(f"    {md_file.relative_to(IOC_ROOT)}: {len(content):,} chars")
        except Exception as e:
            print(f"    WARNING: Failed to read {md_file.name}: {e}")

    # 2. Enriched factory text files (summaries with species, certifications, confidence)
    txt_files = [f for f in IOC_ROOT.glob('**/*.txt') if not should_skip(f)]
    for txt_file in txt_files:
        try:
            content = txt_file.read_text(encoding='utf-8', errors='replace')
            if len(content) < 200:
                continue

            # Split into factory entries (separated by blank lines or dashes)
            entries = content.split('\n\n')
            chunk = []
            chunk_len = 0

            for entry in entries:
                entry = entry.strip()
                if not entry or len(entry) < 30:
                    continue
                if chunk_len + len(entry) > 2000 and chunk:
                    text = f"{EOT}\n[MARKET COMMENTARY]\n" + '\n\n'.join(chunk)
                    documents.append(text)
                    chunk = []
                    chunk_len = 0
                chunk.append(entry)
                chunk_len += len(entry)

            if chunk:
                text = f"{EOT}\n[MARKET COMMENTARY]\n" + '\n\n'.join(chunk)
                documents.append(text)

            print(f"    {txt_file.relative_to(IOC_ROOT)}: {len(content):,} chars")
        except Exception as e:
            print(f"    WARNING: Failed to read {txt_file.name}: {e}")

    # 3. Citation CSV files (evidence snippets with confidence levels)
    import csv
    citation_files = [f for f in IOC_ROOT.glob('**/*citations*.csv') if not should_skip(f)]
    for csv_file in citation_files:
        try:
            with open(csv_file, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                batch = []
                batch_len = 0

                for row in reader:
                    # Build a natural-language description from citation fields
                    parts = []
                    entity = row.get('entity_id') or row.get('entity') or row.get('factory') or ''
                    field = row.get('field', '')
                    value = row.get('value', '')
                    snippet = row.get('evidence_snippet', '')
                    confidence = row.get('confidence', '')

                    if entity and value:
                        line = f"{entity}: {field}={value}"
                        if confidence:
                            line += f" (confidence: {confidence})"
                        if snippet:
                            line += f" — {snippet}"
                        parts.append(line)

                    if parts:
                        entry = ' '.join(parts)
                        if batch_len + len(entry) > 2000 and batch:
                            text = f"{EOT}\n[MARKET COMMENTARY]\n" + '\n'.join(batch)
                            documents.append(text)
                            batch = []
                            batch_len = 0
                        batch.append(entry)
                        batch_len += len(entry)

                if batch:
                    text = f"{EOT}\n[MARKET COMMENTARY]\n" + '\n'.join(batch)
                    documents.append(text)

            print(f"    {csv_file.relative_to(IOC_ROOT)}: citations loaded")
        except Exception as e:
            print(f"    WARNING: Failed to read {csv_file.name}: {e}")

    print(f"  Extracted {len(documents)} IOC chunks total")
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

    # 2. Extract vessel data
    print("\nExtracting vessel registry...")
    all_documents.extend(extract_vessels(client))

    print("Extracting vessel events...")
    all_documents.extend(extract_vessel_events(client))

    print("Extracting fishing effort...")
    all_documents.extend(extract_fishing_effort(client))

    # 3. Extract research docs
    print("\nReading research documents...")
    all_documents.extend(extract_research_docs())

    # 3. Extract IOC seafood map data
    print("\nReading IOC seafood map data...")
    all_documents.extend(extract_ioc_docs())

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
