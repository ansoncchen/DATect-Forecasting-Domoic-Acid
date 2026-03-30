# DATect Ontology Schema

## Overview

Three object types model the operational domain: monitoring sites, ML forecasts, and regulatory actions. This structure supports the full decision loop: **data in -> prediction -> action -> audit trail**.

---

## Object Type 1: HarvestZone

Represents one of the 10 Pacific Coast shellfish monitoring sites. This is the primary operational entity — state fisheries managers think in terms of zones, not data points.

| Property | Type | Indexed | Description |
|----------|------|---------|-------------|
| `zone_id` | `String` | **PK** | Slugified site name (e.g., `long-beach`, `twin-harbors`) |
| `display_name` | `String` | No | Human-readable name (e.g., `Long Beach`) |
| `latitude` | `Double` | No | Site latitude (from `config.SITES`) |
| `longitude` | `Double` | No | Site longitude |
| `state` | `String` | Yes | `WA` or `OR` — enables state-level filtering |
| `current_status` | `String` | Yes | `OPEN`, `RESTRICTED`, or `CLOSED` — updated by AIP Actions |
| `current_risk_level` | `String` | Yes | `Low`, `Moderate`, `High`, or `Extreme` — from latest forecast |
| `last_forecast_date` | `Date` | No | Date of most recent DAForecastResult for this zone |
| `last_da_measurement` | `Double` | No | Most recent non-null `da_raw` value |
| `last_measurement_date` | `Date` | No | Date of the last `da_raw` observation |

**Why this structure:** Zone status and risk level are denormalized onto the zone object (rather than requiring a join to the latest forecast) so that Workshop list views and map widgets can display current state without computed properties. The `state` index enables views like "all closed zones in Washington."

---

## Object Type 2: DAForecastResult

Each row represents one site-week forecast from the ML pipeline. Links back to the HarvestZone it belongs to. This is the time-series object — Workshop time-series widgets bind directly to these.

| Property | Type | Indexed | Description |
|----------|------|---------|-------------|
| `forecast_id` | `String` | **PK** | `{zone_id}_{YYYY-MM-DD}` (e.g., `long-beach_2022-07-18`) |
| `harvest_zone` | `Link -> HarvestZone` | Yes | Foreign key to the parent zone |
| `forecast_date` | `Date` | Yes | The Monday of the forecast week |
| `predicted_da` | `Double` | No | Ensemble DA prediction (ug/g) |
| `predicted_category` | `Integer` | Yes | Risk category: 0=Low, 1=Moderate, 2=High, 3=Extreme |
| `risk_label` | `String` | No | Human-readable category name |
| `spike_probability` | `Double` | No | Sigmoid-based spike probability (0 to 1) |
| `spike_alert` | `Boolean` | Yes | True if spike_prob >= 0.10 OR predicted_da >= 12.0 |
| `confidence_lower` | `Double` | No | q05 lower bound (ug/g) |
| `confidence_median` | `Double` | No | q50 median prediction (ug/g) |
| `confidence_upper` | `Double` | No | q95 upper bound (ug/g) |
| `actual_da` | `Double` | No | Ground truth `da_raw` when available (null if no sample that week) |
| `model_error` | `Double` | No | `|predicted_da - actual_da|` when both exist (null otherwise) |

**Why this structure:** Indexing `spike_alert` and `predicted_category` enables fast Workshop filters like "show me all High/Extreme forecasts this month" or "zones with active spike alerts." The `harvest_zone` link index enables efficient time-series queries per zone. `actual_da` is included for retrospective accuracy tracking — Workshop can show predicted vs. actual overlays.

---

## Object Type 3: HarvestAdvisory

Represents a regulatory action taken in response to a forecast. Created by AIP Actions (zone closures, sampling orders, public advisories). This provides the full audit trail that state agencies require.

| Property | Type | Indexed | Description |
|----------|------|---------|-------------|
| `advisory_id` | `String` | **PK** | Auto-generated UUID or sequential ID |
| `harvest_zone` | `Link -> HarvestZone` | Yes | Which zone this advisory applies to |
| `triggering_forecast` | `Link -> DAForecastResult` | No | The forecast that triggered this action |
| `advisory_type` | `String` | Yes | `CLOSURE`, `SAMPLING_ORDER`, or `PUBLIC_ADVISORY` |
| `severity` | `String` | No | `RESTRICTED` or `CLOSED` (for closures); `ELEVATED` or `URGENT` (for sampling) |
| `issued_date` | `Datetime` | Yes | When the action was taken |
| `effective_date` | `Date` | No | When the advisory takes effect |
| `expiration_date` | `Date` | No | Auto-expiration date (7-14 days depending on type) |
| `status` | `String` | Yes | `ACTIVE`, `EXPIRED`, or `CANCELLED` |
| `issued_by` | `String` | No | `AIP_AUTOMATED` or operator username |
| `rationale` | `String` | No | AI-generated explanation citing forecast values and key drivers |

**Why this structure:** Indexing `status` and `advisory_type` enables operational views like "all active closures" or "pending sampling orders." The `triggering_forecast` link creates traceability from action back to the model prediction that motivated it — critical for regulatory audits and model performance review.

---

## Relationships

```
HarvestZone (10 zones)
    |
    |── 1:N ── DAForecastResult (52 per zone per year)
    |              |
    |              |── 1:N ── HarvestAdvisory (0-3 per forecast)
    |
    |── 1:N ── HarvestAdvisory (direct link for zone-level queries)
```

- **HarvestZone -> DAForecastResult**: One zone has many weekly forecasts. Workshop time-series charts bind to this relationship.
- **DAForecastResult -> HarvestAdvisory**: One forecast can trigger multiple actions (e.g., a High-risk forecast triggers both a closure and a public advisory).
- **HarvestZone -> HarvestAdvisory**: Direct link enables zone-level advisory history without traversing through forecasts.

---

## Indexing Strategy

| Index | Purpose |
|-------|---------|
| `HarvestZone.current_status` | Filter map/list by open/closed status |
| `HarvestZone.current_risk_level` | Sort zones by risk for triage views |
| `DAForecastResult.forecast_date` | Time-range queries for trend analysis |
| `DAForecastResult.spike_alert` | Quick filter to "show me all alerts" |
| `DAForecastResult.(harvest_zone, forecast_date)` | Composite: efficient per-zone time-series |
| `HarvestAdvisory.status` | Active vs. expired advisory filtering |
| `HarvestAdvisory.issued_date` | Recent actions timeline view |
