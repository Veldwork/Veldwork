import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
import time
from pytrends.request import TrendReq

print("=" * 60)
print("  SA ALTERNATIVE DATA RESEARCH")
print("  Signal: Google Trends → JSE Retail Returns")
print("=" * 60)

print("\n[1/4] Fetching Google Trends data via pytrends...")

BRANDS = ['Shoprite', 'Woolworths', 'Mr Price', 'Pick n Pay']

def fetch_trends(brands, max_retries=4):
    for attempt in range(max_retries):
        try:
            print(f"  Attempt {attempt + 1}...")
            pytrends = TrendReq(hl='en-US', tz=120, timeout=(10, 30), retries=2, backoff_factor=0.5)
            pytrends.build_payload(
                kw_list=brands,
                cat=0,
                timeframe='today 5-y',
                geo='ZA',
                gprop=''
            )
            df = pytrends.interest_over_time()
            if df.empty:
                raise ValueError("Google returned empty data")
            if 'isPartial' in df.columns:
                df = df.drop(columns=['isPartial'])
            df.index = df.index.tz_localize(None)
            print(f"  Success! Got {len(df)} rows of data")
            return df
        except Exception as e:
            print(f"  Failed: {e}")
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 15
                print(f"  Waiting {wait} seconds before retry...")
                time.sleep(wait)
    return None

trends_raw = fetch_trends(BRANDS)

if trends_raw is None:
    print("\n  ERROR: Could not fetch Google Trends data after all retries.")
    print("  Try again in a few minutes — Google rate-limits automated requests.")
    exit()

print(f"\n  Brands fetched: {list(trends_raw.columns)}")
print(f"  Date range: {trends_raw.index[0].date()} to {trends_raw.index[-1].date()}")
print(f"  Frequency check (gap between rows): {trends_raw.index[1] - trends_raw.index[0]}")
print(trends_raw.head(5).to_string())

print("\n[2/4] Downloading JSE stock prices...")

TICKERS = {
    'Shoprite':   'SHP.JO',
    'Woolworths': 'WHL.JO',
    'Mr Price':   'MRP.JO',
    'Pick n Pay': 'PIK.JO',
}

brands_available = [b for b in TICKERS.keys() if b in trends_raw.columns]
tickers_to_download = [TICKERS[b] for b in brands_available]

time.sleep(1)

prices_raw = yf.download(
    tickers_to_download,
    period='5y',
    interval='1wk',
    auto_adjust=True,
    progress=False
)['Close']

if len(tickers_to_download) == 1:
    prices_raw = prices_raw.to_frame(name=tickers_to_download[0])

prices_raw.index = prices_raw.index.tz_localize(None)
ticker_to_brand = {v: k for k, v in TICKERS.items()}
prices_raw = prices_raw.rename(columns=ticker_to_brand)
prices_raw = prices_raw.dropna(axis=1, how='all')

print(f"  Downloaded: {list(prices_raw.columns)}")
print(prices_raw.head(5).round(2).to_string())

print("\n[3/4] Calculating weekly returns...")

weekly_returns = prices_raw.pct_change() * 100
weekly_returns = weekly_returns.clip(lower=-30, upper=30)

print("\n[4/4] Aligning data and running correlation analysis...")

# Align trends and prices to the same weekly dates
trends_aligned = trends_raw[brands_available].copy()
returns_aligned = weekly_returns[brands_available].copy()

# Merge on nearest date to handle slight day offsets between the two sources
combined_list = []
for brand in brands_available:
    t = trends_aligned[[brand]].rename(columns={brand: 'searches'})
    r = returns_aligned[[brand]].rename(columns={brand: 'return'})
    merged = pd.merge_asof(
        r.sort_index(),
        t.sort_index().shift(1),  # lag: last week's searches
        left_index=True,
        right_index=True,
        tolerance=pd.Timedelta('4 days'),
        direction='nearest'
    ).dropna()
    merged['brand'] = brand
    combined_list.append(merged)

print(f"  Data rows per brand after alignment:")

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)
print(f"\n  {'Brand':<15} {'N weeks':>8} {'Correlation':>12} {'p-value':>10}   Verdict")
print("  " + "-" * 70)

results = []
for merged in combined_list:
    brand = merged['brand'].iloc[0]
    n = len(merged)
    print(f"  {brand}: {n} weeks of aligned data")
    if n < 10:
        print(f"  {brand:<15} {'N/A':>8} {'N/A':>12} {'N/A':>10}   Not enough data")
        continue
    corr, pval = stats.pearsonr(merged['searches'], merged['return'])
    if pval < 0.01:
        verdict = 'Strong signal *** publishable'
    elif pval < 0.05:
        verdict = 'Good signal ** publishable'
    elif pval < 0.10:
        verdict = 'Weak signal * borderline'
    else:
        verdict = 'No signal found'
    results.append({'brand': brand, 'corr': corr, 'pval': pval, 'verdict': verdict, 'n': n, 'data': merged})
    print(f"  {brand:<15} {n:>8} {corr:>12.3f} {pval:>10.4f}   {verdict}")

print("\n  PLAIN ENGLISH:")
for r in results:
    direction = "go UP" if r['corr'] > 0 else "go DOWN"
    print(f"\n  {r['brand']} (n={r['n']} weeks):")
    print(f"  High searches → share price tends to {direction} next week")
    print(f"  Correlation = {r['corr']:+.3f}, p-value = {r['pval']:.4f}")
    if 'publishable' in r['verdict']:
        print(f"  --> PUBLISHABLE FINDING")
    else:
        print(f"  --> Not significant at conventional thresholds")
print("\n" + "=" * 60)
print("  LAG ANALYSIS (does the signal appear at longer lags?)")
print("=" * 60)

lag_results = {}
for brand in brands_available:
    if brand not in weekly_returns.columns:
        continue
    t = trends_raw[[brand]].rename(columns={brand: 'searches'})
    r = weekly_returns[[brand]].rename(columns={brand: 'return'})
    brand_lags = []
    for lag in range(0, 9):
        merged = pd.merge_asof(
            r.sort_index(),
            t.sort_index().shift(lag),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta('4 days'),
            direction='nearest'
        ).dropna()
        if len(merged) < 10:
            continue
        corr, pval = stats.pearsonr(merged['searches'], merged['return'])
        brand_lags.append({'lag': lag, 'corr': corr, 'pval': pval})
    lag_results[brand] = brand_lags

print(f"\n  {'Lag':<8}", end="")
for brand in lag_results:
    print(f"  {brand:<20}", end="")
print()
print("  " + "-" * 90)

for lag in range(0, 9):
    print(f"  {lag} weeks ", end="")
    for brand, lags in lag_results.items():
        entry = next((x for x in lags if x['lag'] == lag), None)
        if entry:
            stars = '***' if entry['pval'] < 0.01 else ('**' if entry['pval'] < 0.05 else ('*' if entry['pval'] < 0.10 else '   '))
            print(f"  {entry['corr']:+.3f} (p={entry['pval']:.3f}) {stars:<3} ", end="")
    print()

print("\n  *** p<0.01  ** p<0.05  * p<0.10")
if results:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Google Trends → JSE Returns: SA Alternative Data Research', fontsize=13, fontweight='bold')

    ax1 = axes[0]
    brands_plot = [r['brand'] for r in results]
    corrs_plot = [r['corr'] for r in results]
    colors = ['#3B6D11' if c > 0 else '#A32D2D' for c in corrs_plot]
    bars = ax1.bar(brands_plot, corrs_plot, color=colors, width=0.5, edgecolor='white')
    ax1.axhline(0, color='black', linewidth=0.8)
    ax1.axhline(0.3, color='#3B6D11', linewidth=1.2, linestyle='--', alpha=0.7, label='+0.3 threshold')
    ax1.axhline(-0.3, color='#A32D2D', linewidth=1.2, linestyle='--', alpha=0.7)
    ax1.set_title("Correlation: Last Week's Searches → This Week's Return")
    ax1.set_ylabel('Correlation')
    ax1.set_ylim(-0.6, 0.6)
    ax1.legend()
    for bar, val in zip(bars, corrs_plot):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.02 if val >= 0 else -0.06),
                 f'{val:+.3f}', ha='center', fontsize=12, fontweight='bold')

    ax2 = axes[1]
    strongest = max(results, key=lambda r: abs(r['corr']))
    rolling = strongest['data']['searches'].rolling(12).corr(strongest['data']['return'])
    ax2.plot(strongest['data'].index, rolling, color='#185FA5', linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.fill_between(strongest['data'].index, rolling, 0, where=(rolling > 0), color='#3B6D11', alpha=0.25)
    ax2.fill_between(strongest['data'].index, rolling, 0, where=(rolling < 0), color='#A32D2D', alpha=0.25)
    ax2.set_title(f"Rolling 12-Week Correlation: {strongest['brand']}")
    ax2.set_ylabel('Rolling Correlation')
    ax2.set_xlabel('Date')

    plt.tight_layout()
    plt.savefig('results_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n  Chart saved as: results_chart.png")

print("\nDone!")

