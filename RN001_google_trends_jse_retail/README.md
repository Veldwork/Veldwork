# RN001: Google Trends vs JSE Retail Returns

**Finding:** Google search interest does not predict weekly returns for JSE-listed
retailers at any lag from 0 to 8 weeks.

**Data:** 4 retailers (Shoprite, Woolworths, Mr Price, Pick n Pay),
261 weekly observations, June 2021 to June 2026.

**Method:** Pearson correlation between weekly search interest in week t
and forward returns in week t+k, for k = 0 through 8.
Significance threshold p < 0.10.

**Result:** 0 significant coefficients across 36 brand-lag combinations.

## Files

| File | Description |
|------|-------------|
| `analysis.py` | Full Python script |
| `data/google_trends.csv` | Weekly Google Trends data |
| `data/jse_prices.csv` | Weekly adjusted closing prices |

## Full write-up
[Research Note 001: Does Google Search Predict JSE Retail Returns?]
(https://open.substack.com/pub/veldwork/p/does-google-search-predict-jse-retail?r=8nklok&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)
