# AIP Actions Specification

Three operator-triggered actions that close the loop from ML prediction to regulatory response. Each action creates a `HarvestAdvisory` object for audit traceability.

---

## Action 1: Close or Restrict Harvest Zone

**Purpose:** When the model forecasts dangerous DA levels, an operator can immediately restrict or close a harvest zone to protect public health.

**Trigger condition:** `spike_alert == True AND predicted_category >= 2` (High or Extreme risk)

**AIP Function:**
```
closeHarvestZone(
    zone: HarvestZone,
    severity: String,       // "RESTRICTED" or "CLOSED" — auto-suggested based on category
    duration_days: Integer, // Default: 7 for High, 14 for Extreme
    rationale: String       // Auto-generated, operator can edit
)
```

**Logic:**
1. **Auto-suggest severity:**
   - `predicted_category == 3` (Extreme, DA > 40 ug/g) -> `CLOSED`, 14 days
   - `predicted_category == 2` (High, DA 20-40 ug/g) -> `RESTRICTED`, 7 days
   - Operator can override both severity and duration
2. **State changes:**
   - Create `HarvestAdvisory` with `advisory_type = "CLOSURE"` and `status = "ACTIVE"`
   - Update `HarvestZone.current_status` to `"CLOSED"` or `"RESTRICTED"`
   - Set `expiration_date = effective_date + duration_days`
3. **Auto-generated rationale:**
   > "Forecast DA of {predicted_da} ug/g ({risk_label}) at {display_name} on {forecast_date}. Spike probability: {spike_probability * 100}%. 90% confidence interval: [{q05}, {q95}] ug/g. Key drivers: {top_3_features}."

**Guardrails:**
- Zones open for > 90 consecutive days require human approval (safety check against model drift)
- Zones already in `CLOSED` status: action extends the existing closure rather than creating a duplicate
- Cannot close a zone with `predicted_category < 2` without explicit override and written justification

**Notification:** Webhook to state Department of Health (WA DOH or OR OHA based on `zone.state`).

---

## Action 2: Trigger Emergency Sampling

**Purpose:** When the model detects rising risk but recent field data is stale, this action dispatches a priority sampling order so that real measurements can confirm or refute the forecast.

**Trigger condition:** `spike_probability >= 0.30 AND weeks_since_last_measurement >= 2`

Where `weeks_since_last_measurement = (current_date - zone.last_measurement_date).days / 7`

**AIP Function:**
```
triggerEmergencySampling(
    zone: HarvestZone,
    priority: String,       // "ELEVATED" or "URGENT" — auto-suggested
    notes: String           // Optional operator instructions for field team
)
```

**Logic:**
1. **Auto-suggest priority:**
   - `spike_probability >= 0.60` -> `URGENT` (same-day dispatch if possible)
   - `spike_probability >= 0.30` -> `ELEVATED` (within 48 hours)
2. **State changes:**
   - Create `HarvestAdvisory` with `advisory_type = "SAMPLING_ORDER"` and `status = "ACTIVE"`
   - Set `expiration_date = issued_date + 7 days` (auto-expire if no result uploaded)
3. **Sampling instructions** (included in advisory):
   > "Collect razor clam tissue samples from 3 locations within {display_name} zone. Test for domoic acid concentration using HPLC method. Upload results to Foundry dataset within 48 hours of collection."

**Guardrails:**
- Maximum 2 emergency sampling orders per zone per 14-day window (prevent alert fatigue)
- Suppress if a sample was collected within the past 5 days (`zone.last_measurement_date` check)
- If 3+ adjacent zones trigger simultaneously, consolidate into a single regional sampling order

**Integration:** Could push to a field sampling management system or generate a formatted email to the sampling team.

---

## Action 3: Issue Public Advisory

**Purpose:** Publish a consumer-facing advisory warning recreational shellfish harvesters about elevated domoic acid risk, before a formal closure is in place.

**Trigger condition:** `predicted_category >= 1` (Moderate or higher) AND `zone.current_status != "CLOSED"`

**AIP Function:**
```
issuePublicAdvisory(
    zone: HarvestZone,
    advisory_level: String, // "CAUTION", "WARNING", or "DANGER" — auto-mapped
    message: String         // Pre-drafted, operator can edit before publishing
)
```

**Logic:**
1. **Auto-map advisory level from risk category:**
   - `predicted_category == 1` (Moderate, DA 5-20 ug/g) -> `CAUTION`
   - `predicted_category == 2` (High, DA 20-40 ug/g) -> `WARNING`
   - `predicted_category == 3` (Extreme, DA > 40 ug/g) -> `DANGER`
2. **Pre-drafted message template:**
   > "{advisory_level}: Elevated domoic acid levels forecast for {display_name}, {state}. Current model prediction: {predicted_da} ug/g (category: {risk_label}). Recreational shellfish harvesters should check current closures at [state DOH website] before digging. This advisory is based on ML forecasting and may be updated as new sampling data becomes available."
3. **State changes:**
   - Create `HarvestAdvisory` with `advisory_type = "PUBLIC_ADVISORY"` and `status = "ACTIVE"`
   - Set `expiration_date = issued_date + 7 days` (unless renewed by next week's forecast)
   - Does NOT change `HarvestZone.current_status` (advisories are informational, not regulatory)

**Guardrails:**
- Require human review before issuing (operator must click "Confirm" — no auto-publish)
- Maximum 1 advisory per zone per 7-day window (prevent notification fatigue)
- When 3+ contiguous zones trigger advisories, aggregate into a single regional advisory (e.g., "Southern Washington Coast" instead of separate Long Beach + Twin Harbors + Copalis alerts)

**Distribution:** Formatted for state DOH website, social media post, and optional SMS/email subscriber list.

---

## Action Summary Matrix

| Action | Creates Advisory Type | Changes Zone Status | Auto/Manual | Expiration |
|--------|----------------------|--------------------|----|------------|
| Close Harvest Zone | `CLOSURE` | Yes -> `CLOSED`/`RESTRICTED` | Manual approval | 7-14 days |
| Emergency Sampling | `SAMPLING_ORDER` | No | Semi-auto (suppression rules) | 7 days |
| Public Advisory | `PUBLIC_ADVISORY` | No | Manual confirmation | 7 days |
