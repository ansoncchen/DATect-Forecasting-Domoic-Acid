# DATect Demo Script — Palantir Foundry + AIP

**Total runtime: 3 minutes 30 seconds**

---

## [0:00 - 0:30] The Problem — 30 seconds

> "Every spring and summer, harmful algal blooms along the Pacific Coast produce domoic acid — a potent neurotoxin that accumulates in shellfish. If contaminated clams reach consumers, the result is Amnesic Shellfish Poisoning: seizures, permanent memory loss, and in severe cases, death.
>
> State agencies in Washington and Oregon manage this risk by sampling shellfish tissue at 10 coastal zones and closing harvests when domoic acid exceeds the FDA action level of 20 micrograms per gram. But here's the problem: sampling is slow and sparse. Results take days. In the meantime, recreational harvesters are digging clams with no warning.
>
> DATect changes that. It's a machine learning system that forecasts domoic acid concentrations one week ahead using satellite ocean data, climate indices, and historical toxin measurements."

**[SCREEN: Show map of 10 Pacific Coast monitoring sites from Kalaloch, WA to Gold Beach, OR]**

---

## [0:30 - 1:30] The Data Pipeline in Foundry — 60 seconds

> "Here's how the data flows in Foundry."

**[SCREEN: Open the raw dataset `datect-raw` in Foundry — show the CSV with columns: date, site, da_raw (with visible NaN gaps), modis_sst, beuti, pdo, oni, discharge]**

> "Raw data arrives weekly from three sources: NOAA satellite imagery gives us sea surface temperature and fluorescence. USGS river gauges provide Columbia River discharge. And state sampling programs contribute the actual domoic acid measurements — but notice the gaps. In winter, some sites go weeks without a sample."

**[SCREEN: Open the Foundry transform `compute_da_forecast` — show the code briefly]**

> "This Foundry transform handles the messy reality of sparse, irregular environmental data. It computes 25 derived features. The most important: observation-order lag features."

**[Highlight the lag feature section in the code]**

> "Instead of shifting by calendar weeks — which gives you NaN on sparse data — we find the Nth most recent actual observation. If the last sample was 3 weeks ago, that's still lag-1. This is critical when measurements are irregular.
>
> The transform then scores each site-week using a simplified version of our XGBoost and Random Forest ensemble. The model's top feature? Simply the last known DA measurement. Persistence dominates in this domain. But the environmental signals — SST, upwelling, climate indices — give us the lead time to act before the next sample comes in."

**[SCREEN: Show the output dataset `datect-enriched` with predicted_da, risk_label, spike_alert columns visible]**

---

## [1:30 - 2:30] The Operator Dashboard — 60 seconds

> "Now let's see what the operator — a state fisheries manager — actually works with."

**[SCREEN: Open Workshop dashboard. Top section: map widget showing 10 zones color-coded by risk level (green=Low, yellow=Moderate, orange=High, red=Extreme)]**

> "This is the weekly triage view. Each zone is color-coded by risk level. Right now, Long Beach and Twin Harbors are showing High risk — the orange zones in southern Washington."

**[SCREEN: Click on Long Beach zone. Right panel shows: HarvestZone object with current_status=OPEN, current_risk_level=High, last_da_measurement=24.5, last_measurement_date=2022-07-11]**

> "Long Beach is currently open, but the model is forecasting 28 micrograms per gram — that's above the FDA action level. Spike probability is 72%. The confidence interval runs from 14 to 43, so even our lower bound is concerning."

**[SCREEN: Below the zone detail, show a time-series chart of DAForecastResult for Long Beach — predicted_da rising over the past 6 weeks with confidence band widening]**

> "The time-series tells the story: DA has been climbing for six weeks, tracking the summer bloom season. The widening confidence band reflects increasing uncertainty at these elevated levels."

**[SCREEN: Show the object list of DAForecastResult filtered to spike_alert=True — 4 zones currently flagged]**

> "Across all zones, four sites have active spike alerts this week. This is the operator's priority list."

---

## [2:30 - 3:15] AIP Actions — 45 seconds

> "Forecasts are only useful if they lead to action. That's where AIP comes in."

**[SCREEN: On the Long Beach zone detail, click "Close Harvest Zone" action button]**

> "The system recommends closing Long Beach based on the High risk forecast. It's pre-filled: severity CLOSED, duration 14 days, and an auto-generated rationale citing the predicted DA of 28 micrograms per gram and the 72% spike probability. The operator reviews and approves."

**[SCREEN: Show the HarvestAdvisory object being created — status: ACTIVE, type: CLOSURE, issued_by: operator name]**

> "A HarvestAdvisory is created with full traceability — we know which forecast triggered it, who approved it, and when it expires."

**[SCREEN: Navigate to Gold Beach. Show spike_probability rising but last_measurement_date is 3 weeks old]**

> "Meanwhile, Gold Beach hasn't been sampled in three weeks, but the model sees rising risk. The operator triggers Emergency Sampling."

**[SCREEN: Click "Trigger Emergency Sampling" — show the URGENT priority and sampling instructions]**

> "This dispatches a priority sampling order to the field team, with instructions to collect and test within 48 hours. The data gap gets closed, and the model gets fresh ground truth."

---

## [3:15 - 3:30] AIP Logic Assistant — 15 seconds

> "Finally, the AI assistant. An operator asks:"

**[SCREEN: Type into the AIP Logic chat: "Which zones should I be most worried about this week?"]**

> "The assistant responds with a ranked summary — Long Beach at 28 micrograms per gram, Twin Harbors at 22, and flags Gold Beach's data gap. It recommends specific actions for each. This is domain-specific intelligence, grounded in live ontology data, not generic chat."

---

## [3:25 - 3:30] Impact — 5 seconds

> "DATect turns sparse environmental data into timely public health decisions — from satellite to safety advisory, orchestrated in Foundry. The result: faster closures, fewer gaps, and safer shellfish for 2 million recreational harvesters on the Pacific Coast."

---

## Demo Preparation Checklist

- [ ] Load `synthetic_data.csv` as `datect-raw` dataset in Foundry
- [ ] Deploy `foundry_transform.py` and run it to produce `datect-enriched`
- [ ] Create ontology object types: HarvestZone, DAForecastResult, HarvestAdvisory
- [ ] Back HarvestZone and DAForecastResult from the enriched dataset
- [ ] Build Workshop dashboard: map widget + zone detail panel + time-series chart + alert list
- [ ] Configure 3 AIP Actions on HarvestZone objects
- [ ] Deploy AIP Logic assistant with the system prompt from `aip_logic_prompt.md`
- [ ] Pre-seed a few HarvestAdvisory objects to show audit trail
- [ ] Record screen with clean resolution, no notifications
- [ ] Keep narration under 3:30 — practice timing with a stopwatch
