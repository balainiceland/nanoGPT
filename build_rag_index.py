#!/usr/bin/env python3
"""Build RAG vector index for PelagicGPT.

Extracts chunks from Supabase structured data + corpus files,
embeds them with OpenAI text-embedding-3-small, and stores in
Supabase pgvector for retrieval-augmented generation.

Usage:
    python build_rag_index.py              # Preview chunks (no embed)
    python build_rag_index.py --execute    # Embed and store in Supabase
    python build_rag_index.py --execute --clear  # Clear existing + rebuild
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load env from blue-economy-data
BLUE_ECON_ROOT = Path('/Users/Bala_1/dev/blue-economy-data')
load_dotenv(BLUE_ECON_ROOT / '.env')

# Also load agent-playground env for OpenAI key
load_dotenv(Path('/Users/Bala_1/dev/agent-playground/.env'))

SUPABASE_URL = os.environ.get('VITE_SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')

EMBEDDING_MODEL = 'text-embedding-3-small'
EMBEDDING_DIMS = 1536
BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs per request


def get_supabase():
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_openai():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(openai_client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI."""
    resp = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]


# === Chunk Extractors ===

def extract_financial_chunks(db) -> list[dict]:
    """Extract company financials as natural-language chunks."""
    chunks = []

    companies = db.table('companies').select('id, name, ticker, sector').eq('is_active', True).execute()
    company_map = {c['id']: c for c in companies.data or []}

    financials = db.table('company_financials').select(
        'company_id, period_end, period_type, revenue_usd, ebitda_usd, '
        'net_income_usd, harvest_volume_mt, cost_per_kg, employees'
    ).order('period_end', desc=True).execute()

    for f in financials.data or []:
        comp = company_map.get(f['company_id'])
        if not comp:
            continue

        parts = [f"{comp['name']} ({comp['ticker']})"]
        if comp.get('sector'):
            parts[0] += f" — {comp['sector']}"
        parts.append(f"Period: {f['period_end']} ({f.get('period_type', 'annual')})")

        if f.get('revenue_usd'):
            parts.append(f"Revenue: ${f['revenue_usd']:,.0f}")
        if f.get('ebitda_usd'):
            parts.append(f"EBITDA: ${f['ebitda_usd']:,.0f}")
            if f.get('revenue_usd') and f['revenue_usd'] > 0:
                margin = f['ebitda_usd'] / f['revenue_usd'] * 100
                parts.append(f"EBITDA margin: {margin:.1f}%")
        if f.get('net_income_usd'):
            parts.append(f"Net income: ${f['net_income_usd']:,.0f}")
        if f.get('harvest_volume_mt'):
            parts.append(f"Harvest volume: {f['harvest_volume_mt']:,.0f} MT")
        if f.get('cost_per_kg'):
            parts.append(f"Cost per kg: ${f['cost_per_kg']:.2f}")
        if f.get('employees'):
            parts.append(f"Employees: {f['employees']:,}")

        content = '\n'.join(parts)
        chunks.append({
            'content': content,
            'source': 'financial',
            'metadata': {
                'ticker': comp['ticker'],
                'company': comp['name'],
                'period': f['period_end'],
                'sector': comp.get('sector'),
            },
        })

    print(f"  {len(chunks)} financial chunks")
    return chunks


def extract_company_chunks(db) -> list[dict]:
    """Extract company profiles."""
    chunks = []

    companies = db.table('companies').select(
        'name, ticker, exchange, currency, market_cap_usd, country, '
        'headquarters, sector, segments, website, description'
    ).eq('is_active', True).execute()

    for c in companies.data or []:
        parts = [f"{c['name']} ({c['ticker']})"]
        if c.get('sector'):
            parts.append(f"Sector: {c['sector']}")
        if c.get('exchange'):
            parts.append(f"Exchange: {c['exchange']}")
        if c.get('country'):
            parts.append(f"Country: {c['country']}")
        if c.get('headquarters'):
            parts.append(f"HQ: {c['headquarters']}")
        if c.get('market_cap_usd'):
            parts.append(f"Market cap: ${c['market_cap_usd']:,.0f}")
        if c.get('segments'):
            parts.append(f"Segments: {', '.join(c['segments'])}")
        if c.get('description'):
            parts.append(c['description'])

        chunks.append({
            'content': '\n'.join(parts),
            'source': 'company',
            'metadata': {
                'ticker': c['ticker'],
                'company': c['name'],
                'sector': c.get('sector'),
            },
        })

    print(f"  {len(chunks)} company chunks")
    return chunks


def extract_news_chunks(db) -> list[dict]:
    """Extract recent news with AI analysis."""
    chunks = []

    news = db.table('news_items').select(
        'title, summary, ai_analysis, source, published_at, '
        'sentiment_label, company_tags, species_tags'
    ).order('published_at', desc=True).limit(500).execute()

    for n in news.data or []:
        parts = []
        if n.get('title'):
            parts.append(n['title'])
        if n.get('published_at'):
            parts.append(f"Date: {n['published_at'][:10]}")
        if n.get('source'):
            parts.append(f"Source: {n['source']}")
        if n.get('sentiment_label'):
            parts.append(f"Sentiment: {n['sentiment_label']}")
        if n.get('summary'):
            parts.append(n['summary'])
        if n.get('ai_analysis'):
            parts.append(n['ai_analysis'])

        if parts:
            chunks.append({
                'content': '\n'.join(parts),
                'source': 'news',
                'metadata': {
                    'date': (n.get('published_at') or '')[:10],
                    'companies': n.get('company_tags') or [],
                    'species': n.get('species_tags') or [],
                },
            })

    print(f"  {len(chunks)} news chunks")
    return chunks


def extract_signal_chunks(db) -> list[dict]:
    """Extract trading signals with rationale."""
    chunks = []

    signals = db.table('signals').select(
        'signal_name, direction, strength, confidence, '
        'rationale, tradeable_instruments, generated_at, data_snapshot'
    ).eq('is_active', True).execute()

    for s in signals.data or []:
        parts = [f"Signal: {s['signal_name']}"]
        parts.append(f"Direction: {s.get('direction', 'neutral')}")
        if s.get('strength'):
            parts.append(f"Strength: {s['strength']}")
        if s.get('confidence'):
            parts.append(f"Confidence: {s['confidence']}")
        if s.get('tradeable_instruments'):
            parts.append(f"Instruments: {', '.join(s['tradeable_instruments'])}")
        if s.get('rationale'):
            parts.append(s['rationale'])

        chunks.append({
            'content': '\n'.join(parts),
            'source': 'signal',
            'metadata': {
                'signal_name': s['signal_name'],
                'direction': s.get('direction'),
                'tickers': s.get('tradeable_instruments') or [],
            },
        })

    print(f"  {len(chunks)} signal chunks")
    return chunks


def extract_annual_report_chunks() -> list[dict]:
    """Extract chunks from annual reports corpus."""
    chunks = []
    corpus_path = Path(__file__).parent / 'data/pelagic/annual_reports_corpus.txt'

    if not corpus_path.exists():
        print("  No annual_reports_corpus.txt found, skipping")
        return chunks

    content = corpus_path.read_text(encoding='utf-8')
    # Split on EOT markers
    sections = content.split('<|endoftext|>')

    for section in sections:
        section = section.strip()
        if len(section) < 100:
            continue

        # Extract company name from header if present
        lines = section.split('\n')
        company = ''
        for line in lines[:3]:
            if '—' in line:
                company = line.split('—')[0].strip().lstrip('[MARKET COMMENTARY]').strip()
                break

        # Chunk long sections (~1500 chars for better retrieval)
        if len(section) > 2000:
            paragraphs = section.split('\n\n')
            chunk_parts = []
            chunk_len = 0
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if chunk_len + len(para) > 1500 and chunk_parts:
                    chunks.append({
                        'content': '\n\n'.join(chunk_parts),
                        'source': 'annual_report',
                        'metadata': {'company': company},
                    })
                    chunk_parts = []
                    chunk_len = 0
                chunk_parts.append(para)
                chunk_len += len(para)
            if chunk_parts:
                chunks.append({
                    'content': '\n\n'.join(chunk_parts),
                    'source': 'annual_report',
                    'metadata': {'company': company},
                })
        else:
            chunks.append({
                'content': section,
                'source': 'annual_report',
                'metadata': {'company': company},
            })

    print(f"  {len(chunks)} annual report chunks")
    return chunks


def extract_quota_chunks(db) -> list[dict]:
    """Extract quota data as natural-language chunks."""
    chunks = []

    quotas = db.table('quotas').select(
        'species_name, region_name, country, year, season, '
        'tac_mt, caught_mt, utilization_pct, prior_year_tac_mt, change_vs_prior_pct'
    ).order('year', desc=True).limit(500).execute()

    for q in quotas.data or []:
        parts = [f"{q.get('species_name', 'Unknown')} quota — {q.get('region_name', '')} ({q.get('country', '')})"]
        parts.append(f"Year: {q.get('year')}")
        if q.get('tac_mt'):
            parts.append(f"TAC: {q['tac_mt']:,.0f} MT")
        if q.get('caught_mt'):
            parts.append(f"Caught: {q['caught_mt']:,.0f} MT")
        if q.get('utilization_pct'):
            parts.append(f"Utilization: {q['utilization_pct']:.1f}%")
        if q.get('change_vs_prior_pct'):
            parts.append(f"Change vs prior year: {q['change_vs_prior_pct']:+.1f}%")

        chunks.append({
            'content': '\n'.join(parts),
            'source': 'quota',
            'metadata': {
                'species': q.get('species_name'),
                'region': q.get('region_name'),
                'year': q.get('year'),
            },
        })

    print(f"  {len(chunks)} quota chunks")
    return chunks


def main():
    parser = argparse.ArgumentParser(description='Build RAG index for PelagicGPT')
    parser.add_argument('--execute', action='store_true', help='Embed and store in Supabase')
    parser.add_argument('--clear', action='store_true', help='Clear existing chunks before rebuild')
    args = parser.parse_args()

    if args.execute and not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check .env files.")
        sys.exit(1)

    print("=== PelagicGPT RAG Index Builder ===\n")

    # 1. Connect to Supabase
    print("Connecting to Supabase...")
    db = get_supabase()

    # 2. Extract all chunks
    print("\nExtracting chunks...")
    all_chunks = []
    all_chunks.extend(extract_company_chunks(db))
    all_chunks.extend(extract_financial_chunks(db))
    all_chunks.extend(extract_news_chunks(db))
    all_chunks.extend(extract_signal_chunks(db))
    all_chunks.extend(extract_quota_chunks(db))
    all_chunks.extend(extract_annual_report_chunks())

    print(f"\nTotal: {len(all_chunks)} chunks")
    total_chars = sum(len(c['content']) for c in all_chunks)
    print(f"Total characters: {total_chars:,}")

    # Source breakdown
    from collections import Counter
    source_counts = Counter(c['source'] for c in all_chunks)
    for source, count in source_counts.most_common():
        print(f"  {source}: {count}")

    if not args.execute:
        print("\nDry run. Use --execute to embed and store.")
        # Preview a few chunks
        print("\n--- Sample chunks ---")
        for c in all_chunks[:3]:
            print(f"\n[{c['source']}] {c['content'][:200]}...")
        return

    # 3. Clear existing if requested
    if args.clear:
        print("\nClearing existing RAG chunks...")
        db.table('rag_chunks').delete().neq('id', 0).execute()
        print("  Cleared.")

    # 4. Embed and store in batches
    print(f"\nEmbedding {len(all_chunks)} chunks with {EMBEDDING_MODEL}...")
    openai = get_openai()

    inserted = 0
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c['content'] for c in batch]

        # Embed batch
        embeddings = embed_texts(openai, texts)

        # Build rows for insert
        rows = []
        for chunk, embedding in zip(batch, embeddings):
            rows.append({
                'content': chunk['content'],
                'source': chunk['source'],
                'metadata': chunk['metadata'],
                'embedding': embedding,
            })

        # Insert into Supabase
        db.table('rag_chunks').insert(rows).execute()
        inserted += len(rows)

        pct = min(100, inserted * 100 // len(all_chunks))
        print(f"  {inserted}/{len(all_chunks)} ({pct}%)")

        # Rate limiting
        if i + BATCH_SIZE < len(all_chunks):
            time.sleep(0.5)

    print(f"\nDone! Inserted {inserted} chunks into rag_chunks table.")
    print("RAG index is ready for queries.")


if __name__ == '__main__':
    main()
