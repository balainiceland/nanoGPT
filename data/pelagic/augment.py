"""
Generate synthetic training data for PelagicGPT via Claude API.

Generates domain-specific text across three modes:
  - Market commentary (per commodity)
  - Signal narratives (per signal type)
  - Price analysis (per company)

Appends to existing corpus and re-tokenizes.
Target: 2-5M total tokens after augmentation.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import tiktoken
from dotenv import load_dotenv

BLUE_ECON_ROOT = Path('/Users/Bala_1/dev/blue-economy-data')
load_dotenv(BLUE_ECON_ROOT / '.env')

OUTPUT_DIR = Path(__file__).parent
AUGMENTED_FILE = OUTPUT_DIR / 'augmented_corpus.txt'
EOT = '<|endoftext|>'

COMMODITIES = [
    'Atlantic salmon', 'fishmeal', 'fish oil', 'shrimp',
    'tuna', 'cod', 'haddock', 'anchovy', 'sardine',
    'pangasius', 'tilapia', 'crab', 'lobster', 'squid',
]

SIGNAL_TYPES = [
    ('peru_anchovy_collapse', 'Peru anchovy catch falling below seasonal average, triggering fishmeal supply disruption'),
    ('norway_harvest_deviation', 'Norwegian salmon harvest deviating from 5-year average, indicating supply shifts'),
    ('salmon_spot_futures_spread', 'Salmon spot price diverging from Fish Pool forward curve, signaling market expectations'),
    ('el_nino_sst', 'El Nino sea surface temperature anomalies affecting anchovy migration and fishing conditions'),
    ('sea_lice_spike', 'Sea lice treatment costs rising across Norwegian farms, indicating biological stress'),
    ('feed_cost_squeeze', 'Fishmeal and soy prices rising, compressing aquaculture margins'),
    ('china_import_surge', 'Chinese seafood imports accelerating, shifting global demand dynamics'),
    ('quota_announcement', 'ICES or national regulators adjusting fishing quotas, impacting catch volumes'),
    ('currency_dislocation', 'NOK/EUR or NOK/USD moves creating relative value in Norwegian exporters'),
    ('consolidation_signal', 'M&A activity or capacity investment suggesting industry consolidation'),
]

COMPANIES = [
    ('MOWI', 'Mowi ASA', 'Norwegian salmon farmer, world\'s largest producer'),
    ('SALM', 'SalMar ASA', 'Norwegian salmon farmer, cost leader'),
    ('LSG', 'Lerøy Seafood', 'Norwegian integrated seafood group'),
    ('GSF', 'Grieg Seafood', 'Norwegian-Canadian salmon farmer'),
    ('BAKKA', 'Bakkafrost', 'Faroese salmon farmer, premium positioning'),
    ('TU', 'Thai Union', 'Thai tuna and shrimp giant'),
    ('AKVA', 'AKVA Group', 'Aquaculture technology provider'),
    ('AKBM', 'Aker BioMarine', 'Antarctic krill harvester'),
]

VESSEL_SCENARIOS = [
    ('ais_gap_dark_fishing', 'Fleet of trawlers going dark in exclusive economic zone, potential IUU fishing activity'),
    ('port_congestion', 'Major fishing port experiencing vessel congestion, delaying landings and affecting fresh market supply'),
    ('fleet_repositioning', 'Purse seine fleet repositioning from Atlantic to Indian Ocean following tuna migration patterns'),
    ('encounter_transshipment', 'At-sea encounters between fishing vessels and reefer carriers suggesting transshipment activity'),
    ('effort_surge', 'Sudden increase in fishing effort in quota-managed zone, potential race-to-fish before season closure'),
    ('loitering_bunkering', 'Carrier vessels loitering at known bunkering locations, indicating fleet operational patterns'),
    ('fleet_composition_shift', 'Change in vessel type composition in key fishing ground, signaling species targeting shift'),
    ('ais_coverage_anomaly', 'AIS reception anomaly in remote fishing area, complicating catch monitoring and verification'),
]

VESSEL_INTELLIGENCE_PROMPT = """Generate 5 realistic vessel intelligence analysis paragraphs about: "{scenario_name}".
Scenario: {description}
Each should be 80-150 words, written as a maritime intelligence analyst for a seafood-focused hedge fund.
Cover: vessel tracking data (AIS/VMS), fleet behavior patterns, fishing effort metrics,
regulatory implications (IUU, quota compliance), and market impact on seafood supply/prices.
Use specific but plausible details (vessel counts, coordinates, hours, regions like FAO areas).
Separate each paragraph with a blank line. Do NOT number them."""

MARKET_COMMENTARY_PROMPT = """Generate 5 realistic blue economy market commentary paragraphs about {commodity}.
Each should be 80-150 words, written in the style of a hedge fund market analyst.
Cover themes like: price movements, supply/demand dynamics, seasonal patterns,
regulatory impact, trade flows, weather effects, and investment implications.
Use specific but plausible numbers (prices, percentages, volumes).
Separate each paragraph with a blank line. Do NOT number them."""

SIGNAL_NARRATIVE_PROMPT = """Generate 5 realistic signal narrative paragraphs for the trading signal: "{signal_name}".
Signal description: {description}
Each should be 60-120 words, written as a quantitative analyst explaining why a signal fired.
Include: signal strength (0.0-1.0), historical hit rate, expected forward return, time horizon.
Use specific but plausible numbers. Separate with blank lines. Do NOT number them."""

PRICE_ANALYSIS_PROMPT = """Generate 5 realistic price analysis paragraphs for {company_name} ({ticker}).
Company profile: {profile}
Each should be 80-150 words, covering: recent price action, earnings, valuation multiples,
peer comparison, catalyst assessment, and risk factors specific to blue economy investing.
Use specific but plausible financial metrics. Separate with blank lines. Do NOT number them."""


def get_anthropic_client():
    """Create Anthropic client."""
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Cannot generate synthetic data.")
        sys.exit(1)
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)


def generate_texts(client, prompt: str) -> list[str]:
    """Call Claude to generate training texts."""
    try:
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=2048,
            messages=[{'role': 'user', 'content': prompt}],
        )
        text = response.content[0].text
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
        return paragraphs
    except Exception as e:
        print(f"  WARNING: Generation failed: {e}")
        return []


def main():
    print("=== PelagicGPT Data Augmentation ===\n")

    client = get_anthropic_client()
    all_texts = []

    # 1. Market commentary per commodity
    print("Generating market commentary...")
    for commodity in COMMODITIES:
        prompt = MARKET_COMMENTARY_PROMPT.format(commodity=commodity)
        paragraphs = generate_texts(client, prompt)
        for p in paragraphs:
            all_texts.append(f"{EOT}\n[MARKET COMMENTARY]\n{p}")
        print(f"  {commodity}: {len(paragraphs)} paragraphs")
        time.sleep(0.5)  # Rate limit

    # 2. Vessel intelligence
    print("\nGenerating vessel intelligence...")
    for scenario_name, description in VESSEL_SCENARIOS:
        prompt = VESSEL_INTELLIGENCE_PROMPT.format(scenario_name=scenario_name, description=description)
        paragraphs = generate_texts(client, prompt)
        for p in paragraphs:
            all_texts.append(f"{EOT}\n[VESSEL INTELLIGENCE]\nScenario: {scenario_name}\n{p}")
        print(f"  {scenario_name}: {len(paragraphs)} paragraphs")
        time.sleep(0.5)

    # 3. Signal narratives
    print("\nGenerating signal narratives...")
    for signal_name, description in SIGNAL_TYPES:
        prompt = SIGNAL_NARRATIVE_PROMPT.format(signal_name=signal_name, description=description)
        paragraphs = generate_texts(client, prompt)
        for p in paragraphs:
            all_texts.append(f"{EOT}\n[SIGNAL NARRATIVE]\nSignal: {signal_name}\n{p}")
        print(f"  {signal_name}: {len(paragraphs)} paragraphs")
        time.sleep(0.5)

    # 4. Price analysis per company
    print("\nGenerating price analysis...")
    for ticker, name, profile in COMPANIES:
        prompt = PRICE_ANALYSIS_PROMPT.format(ticker=ticker, company_name=name, profile=profile)
        paragraphs = generate_texts(client, prompt)
        for p in paragraphs:
            all_texts.append(f"{EOT}\n[PRICE ANALYSIS]\nTicker: {ticker}\n{p}")
        print(f"  {ticker}: {len(paragraphs)} paragraphs")
        time.sleep(0.5)

    # 5. Save augmented corpus
    corpus = '\n'.join(all_texts) + f'\n{EOT}\n'
    AUGMENTED_FILE.write_text(corpus, encoding='utf-8')
    print(f"\nSaved {len(all_texts)} augmented documents to {AUGMENTED_FILE.name}")

    # 6. Combine with existing corpus and re-tokenize
    print("\nRe-tokenizing combined corpus...")
    existing_train = OUTPUT_DIR / 'train.bin'
    existing_val = OUTPUT_DIR / 'val.bin'

    enc = tiktoken.get_encoding("gpt2")
    augmented_tokens = enc.encode(corpus, allowed_special={EOT})
    print(f"Augmented tokens: {len(augmented_tokens):,}")

    if existing_train.exists():
        # Load existing tokens and combine
        existing = np.fromfile(existing_train, dtype=np.uint16)
        existing_v = np.fromfile(existing_val, dtype=np.uint16)
        all_existing = np.concatenate([existing, existing_v])
        combined = np.concatenate([all_existing, np.array(augmented_tokens, dtype=np.uint16)])
        print(f"Existing tokens: {len(all_existing):,}")
        print(f"Combined tokens: {len(combined):,}")
    else:
        combined = np.array(augmented_tokens, dtype=np.uint16)
        print(f"No existing data found. Using augmented data only.")

    # 7. Re-split and save
    n = len(combined)
    split_idx = int(n * 0.9)
    train_ids = combined[:split_idx]
    val_ids = combined[split_idx:]

    train_ids.tofile(OUTPUT_DIR / 'train.bin')
    val_ids.tofile(OUTPUT_DIR / 'val.bin')

    print(f"\ntrain.bin: {len(train_ids):,} tokens")
    print(f"val.bin:   {len(val_ids):,} tokens")
    print(f"Total:     {len(combined):,} tokens")
    print("\nDone! Ready for training.")


if __name__ == '__main__':
    main()
