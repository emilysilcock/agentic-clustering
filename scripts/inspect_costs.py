"""Inventory cost fields across methods/datasets for the table refactor."""

import json
import glob
import os

for path in sorted(glob.glob("results/predictions/*/banking77/seed=0.meta.json")):
    m = json.load(open(path, encoding="utf-8"))
    method = m.get("method", path.split(os.sep)[-3])
    c = m.get("cost", {})
    print(
        f"{method:<32} usd={c.get('usd', '?')} api={c.get('api_usd', '?')} "
        f"sub={c.get('subscription_usd', '?')} wall={c.get('wall_clock_s', '?')}"
    )
