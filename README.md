# TOPSIS laptop ranking with weight sensitivity analysis

A Streamlit app that ranks 11 laptops with the TOPSIS multi-criteria method and
checks how robust the ranking is to the choice of criterion weights.

![App screenshot](docs/screenshot.png)
<!-- TODO (author): add a screenshot of the running app as docs/screenshot.png -->

## Goal

Choosing a laptop means trading off price, performance, weight, battery and more.
TOPSIS turns these criteria into a single ranking, but the result depends on the
weights, which are subjective. This project answers two questions:

1. Which laptop is best for a given set of weights?
2. How much would the weights have to change for a different laptop to win?

## Method

### TOPSIS

For a decision matrix with alternatives $i$ and criteria $j$:

1. **Vector normalization:** $r_{ij} = x_{ij} / \sqrt{\sum_i x_{ij}^2}$, which removes units (PLN, kg, GB).
2. **Weighting:** weights are normalized to sum to 1, then $v_{ij} = w_j \, r_{ij}$.
3. **Ideal and anti-ideal solutions:** for benefit criteria the ideal is the column
   maximum and the anti-ideal the minimum; for cost criteria the other way round.
4. **Distances:** Euclidean distances $d_i^+$ to the ideal and $d_i^-$ to the anti-ideal.
5. **Closeness coefficient:** $C_i = d_i^- / (d_i^+ + d_i^-) \in [0, 1]$; higher is better.
   Ties share a rank (competition ranking: 1, 1, 3).

### Sensitivity analysis

- **One-at-a-time:** the weight of one criterion goes from 0 to 1 while the other
  weights keep their proportions, so the weights always sum to 1. The app shows
  how each laptop's rank changes and at which weights the winner switches.
- **Monte Carlo:** weight vectors are sampled from a Dirichlet distribution with
  parameters $\alpha = c \cdot w$, where $w$ are the chosen weights and $c$ the
  concentration. The mean of the samples equals $w$; a larger $c$ keeps them
  closer to it. For each laptop the app reports the share of simulations in first
  place and in the top three, the mean rank and the full rank distribution.
  Results are reproducible for a given seed.

## Data

11 laptops, collected manually from [morele.net](https://www.morele.net) on 16.11.2026.
CPU and GPU benchmark scores come from PassMark ([cpubenchmark.net](https://www.cpubenchmark.net)).
Details, units and criterion directions: [data/README.md](data/README.md).

| Criterion | Unit | Direction |
|---|---|---|
| Price | PLN | cost |
| System | 0 = none, 1 = Windows 11 Home, 2 = Windows 11 Pro | benefit |
| RAM | GB | benefit |
| Memory (SSD) | GB | benefit |
| CPU_score | PassMark CPU Mark | benefit |
| GPU_score | PassMark G3D Mark | benefit |
| Display | refresh rate, Hz | benefit |
| Battery | capacity, Wh | benefit |
| Weight | kg | cost |

## How to run

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS / Linux:        source .venv/bin/activate
pip install -r requirements.txt

python src/clean_data.py      # raw data/laptops.csv -> data/laptops_clean.csv
streamlit run src/app.py      # opens the app at http://localhost:8501
pytest                        # run the tests
```

In the app you can use the example data or upload your own CSV (one row per
alternative, an ID column and numeric criteria). The sidebar sets each
criterion's direction and weight; unticking **Include** excludes a criterion
(weight 0). Tabs: **Ranking**, **One-at-a-time**, **Monte Carlo**, **Export**
(ranking as CSV).

## Project structure

```
topsis_sensitivity/
├── data/
│   ├── laptops.csv            raw data as collected (values with units)
│   ├── laptops_clean.csv      numeric data used by the app (generated)
│   └── README.md              sources, collection date, units
├── src/
│   ├── topsis.py              TOPSIS: validation, closeness coefficient, ranking
│   ├── sensitivity.py         one-at-a-time and Monte Carlo analyses, text summary
│   ├── clean_data.py          converts the raw CSV into numbers
│   ├── charts.py              Plotly figures
│   └── app.py                 Streamlit user interface
├── tests/
│   ├── test_topsis.py         hand-calculated cases, method properties, invalid input
│   └── test_sensitivity.py    hand-calculated cases, properties, reproducibility
├── requirements.txt
└── pytest.ini
```

`topsis.py` and `sensitivity.py` are pure functions without Streamlit, so they
can be tested and reused independently of the app. Both analyses use the same
vectorized TOPSIS core as the ranking itself.

## Testing

The tests combine hand-calculated examples (expected values computed by the
author) with properties the method must satisfy: invariance to column scale and
to weight scale, a dominant alternative scoring 1, identical alternatives sharing
a rank, a zero weight being equivalent to dropping the criterion, reproducible
Monte Carlo results for a fixed seed, and readable errors for invalid input.

## Results and conclusions

<!-- TODO (author): your weights, the resulting ranking, how stable the winner is
(one-at-a-time switch points, Monte Carlo shares) and what you conclude. -->

*To be completed by the author.*

## Limitations

- **Weights are subjective.** The sensitivity analysis shows how much they matter
  but cannot choose them.
- **Correlated criteria.** CPU and GPU scores tend to rise together, so performance
  can effectively be counted twice.
- **Rank reversal.** Adding or removing a laptop can change the order of two
  others, because the normalization and the ideal and anti-ideal solutions change.
- **Ordinal and proxy criteria.** `System` is an ordinal code (Pro > Home > none),
  not a measured quantity; battery capacity (Wh) is a proxy for battery life.
- **Snapshot data.** Prices reflect a single day; only weight uncertainty is
  analysed, not uncertainty in the data.
- **Small sample.** 11 laptops from one shop, not a market overview.

---

Code written with AI assistance; method validation, data and analysis by the author.
