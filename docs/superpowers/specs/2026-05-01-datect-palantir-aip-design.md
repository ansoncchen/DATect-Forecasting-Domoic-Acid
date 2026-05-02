# DATect × Palantir AIP — Build Challenge Design Spec
_Date: 2026-05-01_

## Context

DATect is a 21-year ML forecasting system for domoic acid (DA) toxin concentrations along 10 Pacific Coast monitoring sites. The Palantir Build Challenge asks for a functional workflow built in Foundry/AIP that drives real-world operational decisions.

This spec translates DATect into a Palantir operator tool for a **WA Dept of Health biotoxin manager** who decides weekly whether to close coastal beaches to shellfish harvest — a binary decision with ~$1M/week economic stakes and direct public health consequences.

## User + Problem

**Operator**: WA Dept of Health biotoxin manager  
**Decision**: Close or keep open each of 10 coastal beaches to razor-clam / Dungeness crab harvest each week  
**Pain today**: Lab results arrive 3–7 days late; manager stares at spreadsheets with no forward-looking signal  
**What DATect provides**: A 1-week-ahead ensemble forecast (XGBoost + Random Forest) with risk category (Low/Moderate/High/Extreme) and spike probability

## Architecture

### Data Layer (Light mode — pre-computed)
- `datect_beaches.csv` — 10 monitoring sites (beach_id, name, lat, lon, state, status)
- `datect_forecasts.csv` — 1,177 backtest records from DATect's retrospective cache (date, site, predicted_da, risk_label, spike_probability, spike_alert, actual_da)
- `datect_closures.csv` — empty template for HarvestClosure records (written by Action)

### Ontology Layer
| Object Type | Key Properties | Links |
|---|---|---|
| `Beach` | beach_id, beach_name, latitude, longitude, state, status | → Forecast (latest), → HarvestClosure (history) |
| `Forecast` | date, beach_id, predicted_da, risk_label, spike_probability, spike_alert, actual_da | → Beach |
| `HarvestClosure` | closure_id, beach_id, effective_date, expiry_date, reason, issued_by, status | → Beach |

### AIP Logic Agent
**Prompt context**: Agent has read access to Beach and Forecast objects  
**User query**: "Which beaches should I close this week?" or "What's the risk at Newport?"  
**Agent behavior**:
1. Query Forecasts for latest anchor date per beach
2. Filter beaches with spike_alert=YES or risk_label=High/Extreme
3. Return ranked list with predicted_da, risk rationale, uncertainty note
4. Cite DATect ensemble (XGB+RF blend) as source

### Workshop App — "HAB Closure Manager"
**Layout**:
- Header: "WA Dept of Health — Harmful Algal Bloom Monitoring"
- Left panel: AIP Assist chat widget (the AIP Logic agent)
- Center: Map of 10 sites, color-coded by current risk (green/yellow/orange/red)
- Right panel: Site detail on click (forecast chart, risk trend, spike probability gauge)
- Bottom: Closure Action panel — "Issue Closure Order" button → form → writes HarvestClosure

### Action — `IssueClosureOrder`
Inputs: beach_id, effective_date, expiry_date, reason  
Effect: Creates HarvestClosure object, updates Beach.status → CLOSED  
Audit trail: Foundry Ontology write is timestamped and attributed to operator

## Build Sequence (in Foundry)

1. **Data Connection**: Upload 3 CSVs as Foundry Datasets
2. **Ontology Manager**: Create Beach, Forecast, HarvestClosure object types; map columns; link Beach→Forecast
3. **AIP Logic**: Create new Logic, add Ontology tool (Beach + Forecast queries), write system prompt
4. **Workshop**: New app → map widget (Beach objects) + AIP Assist widget + table + Action button
5. **Test**: Ask "Which beaches need attention?" in the agent; click a site on the map; click Issue Closure

## Video Narrative (4 min)

0:00–0:30 — Problem: shellfish closures, economic stakes, current lag in data  
0:30–1:30 — Live demo: Workshop map, ask agent "Which beaches should I close this week?", show agent reasoning  
1:30–2:30 — Click a beach → show forecast detail, spike probability, 19-year backtest track record  
2:30–3:30 — Click "Issue Closure Order" → fill form → HarvestClosure created, map turns red  
3:30–4:00 — Why DATect? R²=0.315 on temporal holdout, 21 years of data, 10 sites, MAE 6.4 µg/g

## Files

| File | Purpose |
|---|---|
| `/tmp/datect_beaches.csv` | 10 sites, ready to upload |
| `/tmp/datect_forecasts.csv` | 1,177 backtest records, ready to upload |
| `/tmp/datect_closures.csv` | Empty schema for HarvestClosure |
| `cache/retrospective/regression_ensemble.json` | Source data |
| `data/processed/final_output.parquet` | Full historical dataset (optional enrichment) |
