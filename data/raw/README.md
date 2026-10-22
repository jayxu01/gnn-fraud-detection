# data/raw

This directory holds the three Elliptic CSVs (not committed to git; already in `.gitignore`).

## Files

| File | Content |
|---|---|
| `elliptic_txs_features.csv` | node features: `txId, time_step, 165 anonymous features` (header-less in this copy; the loader handles both) |
| `elliptic_txs_edgelist.csv` | directed payment flows: `txId1, txId2` |
| `elliptic_txs_classes.csv` | labels: `txId, class` (`unknown` / `1` = illicit / `2` = licit) |

## Download

Source: Kaggle <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>

Option 1 (manual): sign in to Kaggle → Download → unzip → place the three files here.

Option 2 (script): configure the `kaggle` CLI (`~/.kaggle/kaggle.json`), then run:

```bash
python scripts/download_data.py
```

## Validation

The pipeline validates the data on load and fails loudly on mismatch:

- 203,769 nodes, 165 features, time steps 1–49;
- 4,545 illicit and 42,019 licit nodes (a flipped 1/2 label direction swaps these counts and is caught).
