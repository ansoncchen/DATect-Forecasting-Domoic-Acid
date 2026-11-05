# Decision-Support Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the configure-then-run dashboard with a one-screen Forecast page (forecast and
80% conformal range, graded spike risk, latest real measurements, ground data, ranked selectable
metrics, all-sites table), a Track record page fed only by API summaries, and the frontend
cleanup from the duplicate audit.

**Architecture:** One adapter module (`src/services/contract.js`) turns every API response into
plain view models; components and pages consume only those. Decisions and copy live in pure
functions under `src/lib/` with Vitest tests; components stay thin and are checked in the
browser preview against the running API. The approved mockup
(`docs/superpowers/specs/dashboard-mockup/index.html`) is the visual reference for every
Forecast-page component.

**Tech Stack:** React 18, Vite 5, Tailwind 3, trimmed Plotly (`src/utils/Plot.js`), axios,
date-fns; Vitest (new, dev only).

**Spec:** `docs/superpowers/specs/2026-11-03-dashboard-decision-support-design.md`

## Global Constraints

- Start only after the Spark consolidation PR (merged as #23) and the API PR (`feat/dashboard-api`,
  plan `docs/superpowers/plans/2026-11-03-dashboard-api-and-error-band.md`) are merged to `main`; bring this branch up to date with the ccd_host
  `sync_with_base_branch` tool. Never merge to `main`, never force-push.
- Edit `frontend/` only (Task 12's listed docs are the one exception). A missing or differently shaped API field is a message to the API
  session (`local_734451ed-99ad-4a7a-992c-b6e7b13ef45b`), never a backend edit.
- No metric the API reports is computed in the browser. Thresholds (20, 12, 0.10), alert levels,
  category bins and labels, eval windows and the band level come from `GET /api/config`; none
  is a literal in components (the tested sentences in `src/lib/copy.js` are the one exception).
- A value at a threshold counts: spike = actual `>= 20`, regression alert = forecast `>= 12`,
  classifier `>= cutoff`, categories left-closed `[0,5) [5,20) [20,40) [40,inf)`.
- Values print to one decimal; `0 < v < 0.1` prints `<0.1`. Percentages are whole numbers.
- Window labels use an en dash and the inclusive end year: holdout "2022–2023", validation
  "2019–2021".
- Parse API dates with `parseISO` (date-fns), never `new Date('YYYY-MM-DD')` (UTC midnight is the
  previous day in US time zones).
- Colour budget (spec section 7): blue `#2563eb` = model output (forecast, range); red `#b91c1c`
  = the 20 µg/g line and Warning; amber `#b45309` = Watch and caution rules; everything else
  black (`#111827`) and Tailwind grays. Measurements outside the range are hollow, not recoloured.
- No emoji, no tinted cards, no pill badges, no drop shadows, no icons in headings. Panels are
  `bg-white border border-gray-200 rounded-md`; a caution is `border-l-2` amber beside plain
  text; a level is a small square plus its word. Sentence-case labels; `tabular-nums` where
  numbers align.
- Every page works at 375 px wide without horizontal scroll.
- Vitest is the only new dependency. `npm run lint`, `npm run test` and `npm run build` pass
  after every task.
- Spec section 9 calls are settled: weak site = range held `< 0.70` of `>= 10` holdout rows;
  stale = newest measurement `> 21` days before the anchor; default chart range 2 years.

## Review Focus

1. **Values exactly at a threshold** the frontend applies (a forecast of 5.0 or 20.0 for its
   category, a range top of exactly 20 for "reaches the closure level", 21 vs 22 days for
   staleness, coverage exactly 0.70) must land on the documented side. The API decides the alert
   level; the frontend only maps it. Tests in Task 2.
2. **A forecast with no band, or one that failed** (`band: null` with a `band_reason`; or
   `success: false` with `error`, a null value, band and alert, and measurements still filled):
   a no-band forecast shows its value and "No range: <band_reason>"; a failed one shows its
   error; both keep the latest-measurement card and hide the range-dependent tiles. Tests in
   Tasks 1, 2 and 7; checked in Tasks 4 and 8.
3. **Dates in US time zones** must not shift a day (`2023-11-28` shows "28 Nov 2023" in
   `America/Los_Angeles`). The test script pins `TZ`; tests in Task 2.
4. **The last measurement is already ≥ 20**: the spike-risk card shows "Already ≥ 20" and its
   explanation, never Watch or Warning. Tests in Task 2.
5. **No measurement before the anchor, or a slow on-demand forecast** (a cache miss takes
   about 20 s): the latest-measurement card says "No measurement before this forecast", and a
   refetch keeps the previous render at reduced opacity with "Computing forecast; this week is
   not precomputed" instead of blanking the page. Test (`lastMeasurement: null`) in Task 2;
   checked in Tasks 4 and 6.

---

## File map

| Path | Responsibility |
|---|---|
| `src/services/contract.js` | API responses → view models (the only file that knows API field names) |
| `src/services/datect.js` | Fetchers: one function per endpoint, returning view models |
| `src/services/__fixtures__/*.json` | Responses captured from the merged API |
| `src/lib/format.js` | Number, percent, date, age, window-label and ratio wording |
| `src/lib/risk.js` | Category, alert level, staleness, site reliability, data currency |
| `src/lib/copy.js` | Sentences the cards print (exact copy, tested) |
| `src/lib/scale.js` | Piecewise category scale for the forecast bar |
| `src/lib/groundTraces.js` | Plotly traces for the ground-data and predicted-vs-actual charts |
| `src/lib/metricSelection.js` | Metric registry order, defaults, localStorage persistence |
| `src/lib/featureNames.js` | Feature name → plain words |
| `src/lib/csv.js` | CSV export (moved from `utils/retrospective.js`) |
| `src/lib/colors.js` | The four colour constants for Plotly |
| `src/context/ConfigContext.jsx` | Loads `/api/config` + `/api/sites` once; `useConfig()` |
| `src/components/ui/*` | `Panel`, `Caution`, `LevelTag`, `DataCurrencyBanner` |
| `src/components/forecast/*` | `FilterBar`, `ForecastCard`, `CategoryScaleBar`, `SpikeRiskCard`, `LatestMeasurementCard`, `GroundDataChart`, `MetricsPanel`, `metricTiles.jsx`, `AllSitesTable` |
| `src/components/trackrecord/*` | `WindowSelector`, `PerSiteTable`, `PredictedVsActualChart`, `ModelComparison` |
| `src/pages/Forecast.jsx`, `TrackRecord.jsx`, `DataExplorer.jsx`, `About.jsx` | Pages |
| Deleted | `pages/Dashboard.jsx`, `pages/Historical.jsx` (renamed), `utils/retrospective.js`, `utils/constants.js`, `react-datepicker`, `react-select` |

---

### Task 1: API contract, fetchers and the test runner

**Files:**
- Modify: `frontend/package.json` (devDependency `vitest`; script `"test": "TZ=America/Los_Angeles vitest run"`), `frontend/vite.config.js` (`test: { environment: 'node', include: ['src/**/*.test.js'] }`)
- Create: `src/services/contract.js`, `src/services/datect.js`, `src/services/__fixtures__/{config,sites,forecast,forecast_noband,forecast_failed,history,summary,all_sites,track_record}.json`, `src/lib/format.js` (only `windowLabel` here; Task 2 adds the rest)
- Test: `src/services/contract.test.js`, `src/lib/format.test.js` (the `windowLabel` case)

**Interfaces:**
- Consumes (API names from the API session's plan, `docs/superpowers/plans/2026-11-03-dashboard-api-and-error-band.md` on `feat/dashboard-api`):
  - `GET /api/config`: `spike_threshold`, `regression_alert_threshold`, `classifier_alert_cutoff`, `alert_levels [{ name, rule, cutoff, placeholder }]`, `category_bins [0, 5, 20, 40, null]`, `category_labels`, `eval_windows`, `band_level`, `newest_measurement_date`, `latest_forecast_week`, `sites_north_to_south`
  - `POST /api/forecast/enhanced` (regression only): `anchor_date`, `success`, `error` (absent on success), `band { lo, hi, level, n_calibration, p_exceed, category_probabilities } | null`, `band_reason` (null with a band, else why not), `alert { classifier_probability, by_classifier, by_regression, alert, level: 'Quiet' | 'Watch' | 'Warning' | null, level_reason } | null`, `last_measurement { date, value }`, `recent_measurements` (≤ 6, newest last), `source`, `model_version`, and the unchanged `regression` (`predicted_da`, the forecast's value; `feature_importance`; `training_samples`) and `ensemble` blocks, both absent on a failed forecast
  - `GET /api/forecast/all?date=` → `{ date, sites: [{ site, anchor_date, predicted_da, band, alert, last_measurement, source, error }] }` (north to south; took 72 s on demand while the precomputed store is stale)
  - `GET /api/sites/{site}/history?since=2019-01-01` → `{ site, rows }` (track-record row columns plus `alert`, `level`)
  - `GET /api/track-record?window=holdout|validation&site=` → `{ window { name, start, end }, summary, per_site, rows }`; a summary is `{ n, r2, log_r2, mae /* each [v, lo, hi] from the window's chronological JSON, or null (Cannon Beach holdout) */, models { ensemble | naive | linear: { r2, mae } }, band { coverage, coverage_at_onsets, median_width, inside, n_banded, onsets_inside }, onsets { n, caught, quiet, quiet_alarmed }, alarms { n, followed_by_jump, false_below_5, false_5_to_10, false_near_miss }, levels { Quiet|Watch|Warning: { weeks, followed_by_jump } } }` (holdout false alarms 21 / 18 / 48, validation 23 / 13 / 37)
  - `GET /api/metrics/summary`: keeps `overall`, `per_site` (`n`, `n_spikes`, `r2`/`mae` as `[v, lo, hi]`, `r2_log1p`, `spike_recall`, `spike_precision`, `spike_f2`) and `multi_seed`; adds `reliability { holdout | validation: { summary, per_site } }` and `source`
- Produces (JSDoc typedefs in `contract.js`; every later task uses these names, never the API's):
  - `Config = { spikeThreshold, regressionAlert, classifierCutoff, levels: [{ name, cutoff, placeholder }], categories: [{ lo, hi, label }] /* null bin → Infinity; label is the name before ' (' in category_labels ('Low (<5)' → 'Low') */, windows: { validation: Window, holdout: Window }, bandLevel, newestMeasurement, latestWeek, sites: string[] /* north to south */, dateRange: { min, max } }`; `Window = { name, start, end, label }`
  - `Alert = { probability, byClassifier, byRegression, fired, level: 'Quiet' | 'Watch' | 'Warning' | null, reason: string | null }`
  - `Forecast = { site, date, anchorDate, error: string | null, value: number | null, band: { lo, hi, nCalibration } | null, bandReason: string | null, categoryChances: number[4] | null, bandPExceed: number | null, alert: Alert | null, lastMeasurement: { date, value } | null, recentMeasurements: [{ date, value }], parts: { xgb, rf, naive, weights } | null, drivers: [{ feature, importance }], provenance: { source, modelVersion, trainingSamples } }`
  - `HistoryRow = { date, anchorDate, actual, predicted, lo, hi, probability, fired, level }`
  - `Stat = { value, ci: [lo, hi] | null }` (the API sends `r2`/`mae` as a number or `[v, lo, hi]`)
  - `Reliability = { n, r2: Stat | null, logR2: Stat | null, mae: Stat | null, models: { ensemble, naive, linear: { r2, mae } } | null, band: { coverage, onsetCoverage, medianWidth, inside, nBanded, onsetsInside }, onsets: { n, caught, quiet, quietAlarmed }, alarms: { n, followed, falseBelow5, false5To10, falseNearMiss }, levels: { [levelName]: { weeks, followed } } }`
  - `Headline = { n, nSpikes, r2: Stat, logR2: Stat, mae: Stat, spike: { recall, precision, f2 } }` (from the existing `overall` / `per_site`)
  - `Summary = { source, headline: { overall: Headline, perSite: { [site]: Headline } }, reliability: { holdout: { overall: Reliability, perSite: { [site]: Reliability } }, validation: {...} } }`
  - `TrackRecord = { window: Window, overall: Reliability, perSite: { [site]: Reliability }, rows: [{ date, site, actual, ensemble, naive, linear, lo, hi, pExceed, probability, fired }] }` (these rows carry no level)
  - `toConfig(rawConfig, rawSites) → Config`, `toForecast(raw) → Forecast`, `toAllSites(raw) → Forecast[]` (fields the endpoint lacks are `null` / `[]`), `toHistory(raw) → HistoryRow[]`, `toSummary(raw) → Summary`, `toTrackRecord(raw) → TrackRecord`
  - `datect.js`: `getConfig()`, `getForecast(site, date)`, `getAllSites(date)`, `getSiteHistory(site)`, `getSummary()`, `getTrackRecord(windowName, site?)`; each returns a Promise of the view model and rejects with `Error(message)` carrying the API's `detail`.

- [ ] **Step 1: Fixtures are already in place.** The controller recaptured them from `main` at `f89b87b6` with no differences from the committed files; do not recapture (the API may not be running for you). Read them to see the real shapes.
- [ ] **Step 2: Write the failing tests** in `contract.test.js`:

```js
it('maps a forecast with its band and alert parts', () => {
  const f = toForecast(forecastFixture)
  expect(f.site).toBe('Newport')
  expect(f.value).toBeCloseTo(forecastFixture.regression.predicted_da)
  expect(f.error).toBeNull()
  expect(f.band.lo).toBeLessThan(f.value)
  expect(f.band.hi).toBeGreaterThan(f.value)
  expect(f.categoryChances.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  expect(typeof f.alert.byClassifier).toBe('boolean')
  expect(['Quiet', 'Watch', 'Warning', null]).toContain(f.alert.level)
  expect(f.lastMeasurement.date <= f.anchorDate).toBe(true)
  expect(f.recentMeasurements.length).toBeLessThanOrEqual(6)
  expect(f.recentMeasurements.at(-1)).toEqual(f.lastMeasurement)
})
it('leaves band fields null when the API sends no band', () => {
  const f = toForecast(forecastNoBandFixture)
  expect(f.band).toBeNull(); expect(f.categoryChances).toBeNull(); expect(f.bandPExceed).toBeNull()
  expect(f.bandReason).toBe('fewer than 30 earlier forecast errors')
})
it('keeps measurements on a failed forecast', () => {
  const f = toForecast(forecastFailedFixture)
  expect(f.error).toBeTruthy(); expect(f.value).toBeNull(); expect(f.alert).toBeNull(); expect(f.band).toBeNull()
  expect(f.anchorDate).toBe('2002-12-30')
  expect(f.lastMeasurement).toBeTruthy(); expect(f.recentMeasurements).toHaveLength(6)
})
it('builds config thresholds, levels, categories and site order', () => {
  const c = toConfig(configFixture, sitesFixture)
  expect(c.spikeThreshold).toBe(20); expect(c.regressionAlert).toBe(12); expect(c.classifierCutoff).toBe(0.1)
  expect(c.levels.find(l => l.name === 'Warning')).toMatchObject({ cutoff: 0.35, placeholder: false })
  expect(c.windows.holdout.label).toBe('2022–2023'); expect(c.windows.validation.label).toBe('2019–2021')
  expect(c.categories.map(k => k.lo)).toEqual([0, 5, 20, 40]); expect(c.categories.at(-1).hi).toBe(Infinity)
  expect(c.sites[0]).toBe('Kalaloch'); expect(c.sites.at(-1)).toBe('Gold Beach')
})
it('maps the holdout reliability summary', () => {
  const r = toSummary(summaryFixture).reliability.holdout
  expect(r.overall.onsets).toEqual({ n: 27, caught: 19, quiet: 292, quietAlarmed: 87 })
  expect(r.overall.alarms).toEqual({ n: 106, followed: 19, falseBelow5: 21, false5To10: 18, falseNearMiss: 48 })
  expect(r.overall.levels.Watch).toEqual({ weeks: 68, followed: 9 })
  expect(r.overall.levels.Warning).toEqual({ weeks: 38, followed: 10 })
  expect(r.perSite.Newport.band).toMatchObject({ inside: 23, nBanded: 48 })
  expect(r.perSite.Newport.onsets).toEqual({ n: 6, caught: 5, quiet: 24, quietAlarmed: 8 })
  expect(r.perSite['Cannon Beach'].n).toBe(2); expect(r.perSite['Cannon Beach'].r2).toBeNull()
})
it('normalizes r2 sent as a number or as [v, lo, hi]', () => {
  expect(toStat(0.516)).toEqual({ value: 0.516, ci: null })
  expect(toStat([0.516, 0.308, 0.654])).toEqual({ value: 0.516, ci: [0.308, 0.654] })
  expect(toStat(null)).toBeNull()
})
```

- [ ] **Step 3: Run** `npm run test` — Expected: FAIL, `toForecast is not a function`.
- [ ] **Step 4: Implement** `windowLabel(start, end)` (end exclusive: `'2022-01-01', '2024-01-01'` → `'2022–2023'`), `toStat`, the six adapters and `datect.js` (axios instance from `services/api.js`). Sites come from `sites_north_to_south`. All fields above are confirmed in the API plan (its commit `7ec00e19`); `/api/forecast/enhanced` keeps `regression` (`feature_importance`, `training_samples`) and `ensemble` (`xgb_prediction`, `rf_prediction`, `naive_prediction`, `ensemble_prediction`, `ensemble_weights`) at their current paths; only `classification` is gone.
- [ ] **Step 5: Run** `npm run test` — Expected: PASS. Run `npm run lint` — Expected: no errors.
- [ ] **Step 6: Commit** `feat(frontend): API contract adapters, fetchers and Vitest`.

### Task 2: Pure decision and copy helpers

**Files:**
- Modify: `src/lib/format.js` (from Task 1)
- Create: `src/lib/risk.js`, `src/lib/copy.js`, `src/lib/scale.js`, `src/lib/colors.js`
- Test: `src/lib/format.test.js` (extend), `src/lib/risk.test.js`, `src/lib/copy.test.js`, `src/lib/scale.test.js`

**Interfaces:**
- Consumes: `Config`, `Forecast`, `Reliability`, `Summary` (Task 1).
- Produces:
  - `format.js` (beside Task 1's `windowLabel`): `formatValue(v) → string`, `formatPct(p) → string`, `formatDate(iso) → '28 Nov 2023'`, `daysBetween(isoA, isoB) → int`, `formatAge(days) → '5 days' | '3 weeks' | '15 months'`, `ratioWords(k, n) → string`.
  - `risk.js`: `STALE_DAYS = 21`, `WEAK_COVERAGE = 0.7`, `MIN_ROWS = 10`; `categoryOf(v, categories) → category`, `reachesThreshold(band, threshold) → bool`, `levelKey(alert) → 'already' | 'quiet' | 'watch' | 'warning' | 'unavailable'` (the API decides the level; this only maps it), `isStale(lastDate, anchorDate) → bool`, `siteReliability(reliability) → 'weak' | 'ok' | 'too-few'`, `dataCurrencyNotice(newestIso, todayIso) → string | null`.
  - `copy.js`: `reliabilityText(site, reliability, overall, holdoutLabel)`, `levelRecordText(levelName, { weeks, followed }, holdoutLabel)`, `trackRecordText(overall, holdoutLabel)`, `siteRecordText(site, onsets)`, `bandVsAlertText(forecast)`.
  - `scale.js`: `categoryScaleX(v, categories) → 0..100` (each category a quarter; the open last category is drawn to `2 × lo`).
  - `colors.js`: `BLUE`, `RED`, `AMBER`, `INK` (values in Global Constraints).

- [ ] **Step 1: Write the failing tests** (`cats` is `Config.categories` for bins `[0, 5, 20, 40, null]`; `newport` and `overall` are holdout `Reliability` objects with the values shown in the copy):

```js
// format.test.js
expect(formatValue(0)).toBe('0.0'); expect(formatValue(0.05)).toBe('<0.1'); expect(formatValue(17.46)).toBe('17.5')
expect(formatDate('2023-11-28')).toBe('28 Nov 2023')          // runs under TZ=America/Los_Angeles
expect(formatAge(1)).toBe('1 day'); expect(formatAge(5)).toBe('5 days'); expect(formatAge(20)).toBe('3 weeks'); expect(formatAge(475)).toBe('16 months')
expect(ratioWords(19, 27)).toBe('about 7 in 10'); expect(ratioWords(19, 106)).toBe('about 1 in 6')
expect(ratioWords(9, 68)).toBe('about 1 in 8'); expect(ratioWords(0, 213)).toBe('never')
// risk.test.js
expect(categoryOf(5, cats).label).toBe('Moderate'); expect(categoryOf(20, cats).label).toBe('High')
expect(reachesThreshold({ lo: 2, hi: 20 }, 20)).toBe(true); expect(reachesThreshold(null, 20)).toBe(false)
expect(levelKey({ level: 'Watch', reason: null })).toBe('watch')
expect(levelKey({ level: 'Warning', reason: null })).toBe('warning')
expect(levelKey({ level: 'Quiet', reason: null })).toBe('quiet')
expect(levelKey({ level: null, reason: 'already_at_or_above_threshold' })).toBe('already')
expect(levelKey({ level: 'Watch', reason: 'already_at_or_above_threshold' })).toBe('already')
expect(levelKey({ level: null, reason: null })).toBe('unavailable'); expect(levelKey(null)).toBe('unavailable')
expect(isStale('2023-10-30', '2023-11-20')).toBe(false); expect(isStale('2023-10-29', '2023-11-20')).toBe(true)
expect(isStale(null, '2023-11-20')).toBe(true)                                               // no measurement before the anchor
expect(siteReliability({ band: { coverage: 0.479, nBanded: 48 } })).toBe('weak')
expect(siteReliability({ band: { coverage: 0.70, nBanded: 40 } })).toBe('ok')
expect(siteReliability({ band: { coverage: 1, nBanded: 2 } })).toBe('too-few'); expect(siteReliability(undefined)).toBe('too-few')
expect(dataCurrencyNotice('2023-11-30', '2026-11-03')).toBe('Forecasts use data through 30 Nov 2023; they are not live.')
expect(dataCurrencyNotice('2026-10-20', '2026-11-03')).toBeNull()
// copy.test.js (Newport, overall coverage 0.802)
expect(reliabilityText('Newport', newport, overall, '2022–2023')).toBe(
  'Less reliable here. In 2022–2023 the range held 23 of 48 measurements at Newport (48%), against 80% across all sites. Expect values outside it.')
expect(levelRecordText('Watch', { weeks: 68, followed: 9 }, '2022–2023')).toBe(
  'Watch weeks in 2022–2023 were followed by a jump 9 of 68 times (about 1 in 8).')
expect(trackRecordText(overall, '2022–2023')).toBe(          // overall.alarms.falseNearMiss = 48
  '2022–2023, all sites: caught 19 of 27 jumps (about 7 in 10); 19 of 106 alarms were followed by one (about 1 in 6). 48 of the 87 false alarms were near misses at 10–20 µg/g.')
expect(siteRecordText('Newport', { n: 6, caught: 5, quiet: 24, quietAlarmed: 8 })).toBe(
  'Newport: 5 of 6 jumps caught; 5 of 13 alarms followed by one.')
expect(bandVsAlertText({ bandPExceed: 0.064, alert: { probability: 0.223 } })).toMatch(/^Not the alert\. The range puts ≥ 20 at 6%; the alert's classifier says 22%\./)
// scale.test.js
expect(categoryScaleX(0, cats)).toBe(0); expect(categoryScaleX(5, cats)).toBe(25)
expect(categoryScaleX(20, cats)).toBe(50); expect(categoryScaleX(200, cats)).toBe(100)
```

- [ ] **Step 2: Run** `npm run test` — Expected: FAIL on missing modules.
- [ ] **Step 3: Implement.** `ratioWords`: `'never'` when `k = 0`; `'about N in 10'` (N = round(10k/n)) when `k/n >= 0.3`; else `'about 1 in ' + round(n/k)`. `levelKey`: `'already'` when `reason === 'already_at_or_above_threshold'`, else the lower-cased `level`, else `'unavailable'` (the card then says "Spike classifier unavailable for this week"). `reliabilityText` prints the API's `band.inside` and `band.nBanded`, never a recount, and covers the three `siteReliability` cases with the mockup's copy (too few: "Only N measurements here in 2022–2023, too few to check the range at this site (all sites: 80%)."; ok: "In 2022–2023 the range held K of N measurements at Site (P%; all sites 80%).").
- [ ] **Step 4: Run** `npm run test` — Expected: PASS; `npm run lint` — no errors.
- [ ] **Step 5: Commit** `feat(frontend): decision and copy helpers with tests`.

### Task 3: Shell: config context, routes, navigation, UI primitives

**Files:**
- Create: `src/context/ConfigContext.jsx`, `src/components/ui/{Panel,Caution,LevelTag,DataCurrencyBanner}.jsx`, `src/pages/Forecast.jsx` (placeholder heading only), `src/pages/TrackRecord.jsx` (placeholder)
- Modify: `src/App.jsx`, `src/components/Navigation.jsx`, `src/index.css` (body `tabular-nums` utility class `.num`)

**Interfaces:**
- Consumes: `getConfig` (Task 1), `dataCurrencyNotice` (Task 2), colours (Task 2).
- Produces: `<ConfigProvider>` and `useConfig() → { config: Config | null, error: string | null }`; `<Panel n?>` (white bordered section), `<Caution>` (amber left rule), `<LevelTag level config />` (square + label; `'already'` renders "Already ≥ <config.spikeThreshold>"; `'quiet'` renders a gray square and "Quiet"; `'unavailable'` renders "Unavailable").

- [ ] **Step 1: Implement** routes `/` Forecast, `/track-record` Track record, `/data` Data explorer (renders the existing `Historical` page until Task 11 renames it), `/about` About, and `/historical` → `<Navigate to="/data" replace />`. Navigation: wordmark unchanged; links as plain text, the active one `bg-gray-100 text-gray-900`; no icons. `DataCurrencyBanner` renders `dataCurrencyNotice(config.newestMeasurement, today)` as one gray line under the nav on every page. Config load failure shows "Could not load settings from the API: <detail>" and no page body.
- [ ] **Step 2: Verify.** `npm run lint && npm run test && npm run build` pass. With `datect-api` and `datect-frontend` running, each route renders its heading, `/historical` lands on `/data`, and the banner reads "Forecasts use data through … ; they are not live."
- [ ] **Step 3: Commit** `feat(frontend): config context, routes and UI primitives`.

### Task 4: Forecast page, filter bar and forecast card

**Files:**
- Create: `src/components/forecast/{FilterBar,ForecastCard,CategoryScaleBar}.jsx`
- Modify: `src/pages/Forecast.jsx`

**Interfaces:**
- Consumes: `useConfig`, `getForecast`, `getSummary`, `formatValue`, `formatDate`, `categoryOf`, `categoryScaleX`, `reliabilityText`, `siteReliability`.
- Produces: `Forecast.jsx` state `{ site, week, forecast, summary, loading, error }` passed to every card; `site` persists in `localStorage['datect.site']` (try/catch; default `config.sites[0]`); `week` defaults to `config.latestWeek`.

- [ ] **Step 1: Implement `FilterBar`**: native `<select>` of `config.sites` with " (less reliable)" / " (little recent data)" from `siteReliability`; native `<input type="date">` with `min = dateRange.min` (weeks before the first band render "No range", spec section 9), `max = config.latestWeek`, snapping any pick to that week's Monday (`startOfWeek(d, { weekStartsOn: 1 })`), labelled "Forecast week"; "Uses data through <anchor>"; right-aligned "Model: ensemble (XGBoost + Random Forest)".
- [ ] **Step 2: Implement `ForecastCard` and `CategoryScaleBar`** to the mockup's card 2: value, "80% range lo–hi µg/g", "In the <Category (lo–hi)> category" plus "; the range reaches the <spikeThreshold> µg/g closure level" when `reachesThreshold(band, spikeThreshold)`, the scale bar (four gray segments, band in blue at 18% fill, forecast tick blue, 20 line red), then `reliabilityText` (wrapped in `<Caution>` when weak). With `band === null`: "No range: <bandReason>" and no band on the bar. With `error`: the error in a `<Caution>` in place of the value and bar.
- [ ] **Step 3: Loading and errors.** While a request is in flight, keep the previous cards at `opacity-60`; after 3 s without a response show "Computing forecast; this week is not precomputed". On error show the API detail in a `<Caution>` above the cards.
- [ ] **Step 4: Verify in the preview** for Newport (latest week), Gold Beach and a no-band week: the card matches the mockup's card 2; the Newport reliability line reads "held 23 of 48 … (48%)"; 375 px has no horizontal scroll (`document.documentElement.scrollWidth === innerWidth`). Screenshot Newport.
- [ ] **Step 5: Commit** `feat(frontend): forecast page, filter bar and forecast card`.

### Task 5: Spike-risk card

**Files:**
- Create: `src/components/forecast/SpikeRiskCard.jsx`
- Modify: `src/pages/Forecast.jsx`

**Interfaces:**
- Consumes: `levelKey`, `levelRecordText`, `trackRecordText`, `siteRecordText`, `formatPct`, `LevelTag`, `Config.levels`, and `Summary.reliability.holdout` (`overall.levels`, `overall.onsets`, `overall.alarms`, `perSite[site].onsets`).

- [ ] **Step 1: Implement** to the mockup's card 3: header "Spike risk" with `<LevelTag>`; the probability (`formatPct`) and "chance of a jump to ≥ 20 µg/g next week"; a gray track with one tick per `Config.levels` entry above Quiet, labelled "10% Watch" / "35% Warning" from `cutoff` and `name`, and a black marker at the probability; two condition rows (● holds / ○ does not) for classifier ≥ cutoff and forecast ≥ regression alert, taken from `alert.byClassifier` / `alert.byRegression`; `levelRecordText` for the current level (omitted for `'already'`, replaced by "The last measurement (v on date) is already at or above 20. This alert is for jumps from below 20; follow the forecast and the measurements."); "Measured lately:" with the last five `recentMeasurements` values; a rule, then `trackRecordText` and `siteRecordText`, then "The Warning cutoff is a placeholder until the alert experiments set the levels." in gray-400 for each level with `placeholder: true`. `'unavailable'` shows the probability as "—" and "Spike classifier unavailable for this week".
- [ ] **Step 2: Verify in the preview**: Newport reads Watch with "followed by a jump 9 of 68 times"; Gold Beach shows its level from config; a week whose last measurement is ≥ 20 shows "Already ≥ 20". Screenshot.
- [ ] **Step 3: Commit** `feat(frontend): graded spike-risk card`.

### Task 6: Latest-measurement card

**Files:**
- Create: `src/components/forecast/LatestMeasurementCard.jsx`
- Modify: `src/pages/Forecast.jsx`

- [ ] **Step 1: Implement** to the mockup's card 4: value, "date, <age> before the forecast was made" (`daysBetween(last.date, anchorDate)`, `formatAge`); `<Caution>` "No recent sample. Without one the forecast leans on ocean conditions and this site's history; give it less weight." when `isStale`; "Last N measurements" rows (date, gray bar scaled to `max(values, spikeThreshold)`, dark bar when `>= spikeThreshold`, red 1px tick at 20, value). `lastMeasurement === null`: "No measurement before this forecast".
- [ ] **Step 2: Verify in the preview** with Cannon Beach (stale caution shows) and Newport. Screenshot.
- [ ] **Step 3: Commit** `feat(frontend): latest-measurement card`.

### Task 7: Ground-data chart

**Files:**
- Create: `src/lib/groundTraces.js`, `src/components/forecast/GroundDataChart.jsx`
- Test: `src/lib/groundTraces.test.js`
- Modify: `src/pages/Forecast.jsx`

**Interfaces:**
- Consumes: `getSiteHistory`, `HistoryRow` (its `level` comes from the API), `Forecast`, colours.
- Produces: `buildGroundTraces(rows, forecast | null, config, { range, showAfter }) → { traces, layout }`, used again by Task 10 with `getSiteHistory` rows limited to the window.

- [ ] **Step 1: Write the failing tests:**

```js
it('breaks line and band across gaps longer than 21 days', () => {
  const { traces } = buildGroundTraces(rows(['2023-01-02', '2023-01-09', '2023-03-06']), fc, cfg, { range: 'all' })
  expect(traces.find(t => t.name === 'Forecast, made a week ahead').x).toEqual(['2023-01-02', '2023-01-09', null, '2023-03-06'])
})
it('draws measurements outside the range hollow', () => {
  const { traces } = buildGroundTraces([row({ actual: 30, lo: 1, hi: 10 }), row({ actual: 5, lo: 1, hi: 10 })], fc, cfg, { range: 'all' })
  expect(traces.find(t => t.name === 'Measurement outside the range').x).toHaveLength(1)
  expect(traces.find(t => t.name === 'Measurement outside the range').marker.color).toBe('white')
})
it('keeps rows after the anchor out unless showAfter', () => {
  const after = row({ date: '2023-12-04' })                 // fc.anchorDate is 2023-11-20
  expect(allX(buildGroundTraces([after], fc, cfg, { range: 'all' }).traces)).not.toContain('2023-12-04')
  expect(allX(buildGroundTraces([after], fc, cfg, { range: 'all', showAfter: true }).traces)).toContain('2023-12-04')
})
it('adds next week with an asymmetric error bar, none without a band', () => {
  const nw = buildGroundTraces([], fc, cfg, { range: '2y' }).traces.find(t => t.name === 'Next week')
  expect(nw.error_y.array[0]).toBeCloseTo(fc.band.hi - fc.value)
  expect(buildGroundTraces([], { ...fc, band: null }, cfg, { range: '2y' }).traces.find(t => t.name === 'Next week').error_y).toBeUndefined()
})
it('omits next week for a failed forecast and for no forecast at all', () => {
  expect(buildGroundTraces([], { ...fc, value: null, band: null }, cfg, { range: '2y' }).traces.find(t => t.name === 'Next week')).toBeUndefined()
  const { traces } = buildGroundTraces([row({ date: '2023-12-04' })], null, cfg, { range: 'all' })  // Track record: no forecast, no anchor cut
  expect(traces.find(t => t.name === 'Next week')).toBeUndefined()
  expect(allX(traces)).toContain('2023-12-04')
})
```

- [ ] **Step 2: Run** `npm run test` — Expected: FAIL.
- [ ] **Step 3: Implement** to the mockup's chart: trace names and styles as in the mockup (`80% range` fill `rgba(37,99,235,0.12)`, forecast line blue 1.5 px, measurements black, outside hollow, Watch/Warning ticks on a 5% strip `yaxis2`, dotted red 20 line labelled "20 µg/g" at the right), legend below with `traceorder: 'normal'`, ranges `6m | 1y | 2y | all` counted back from the forecast week (`all` = every returned row). A vertical gray line at the anchor appears when `showAfter`.
- [ ] **Step 4: Run** `npm run test` — Expected: PASS.
- [ ] **Step 5: Implement `GroundDataChart`**: panel title "Measurements and forecasts, <site>", the mockup's explanatory paragraph with window labels from config, the segmented range control (default `2y`), and a "Show what happened" checkbox shown only when the selected week is before `config.latestWeek`.
- [ ] **Step 6: Verify in the preview** for Newport at 2 yr and since 2019; hover shows date, measured, forecast and range, inside/outside, classifier %. Screenshot.
- [ ] **Step 7: Commit** `feat(frontend): ground-data chart with the conformal band`.

### Task 8: Metrics panel

**Files:**
- Create: `src/lib/metricSelection.js`, `src/lib/featureNames.js`, `src/components/forecast/{MetricsPanel,metricTiles}.jsx`
- Test: `src/lib/metricSelection.test.js`, `src/lib/featureNames.test.js`
- Modify: `src/pages/Forecast.jsx`

**Interfaces:**
- Produces: `METRICS` (ordered `[key, label]` pairs: `alert` Alert track record, `coverage` Range reliability, `mae` Typical error, `cats` Category chances, `r2` Fit (R²), `line` Spike detection by the forecast, `parts` Model parts, `drivers` Top drivers, `width` Range width, `prov` Data and model), `DEFAULT_METRICS = ['alert', 'coverage', 'mae', 'cats']`, `loadMetricSelection(storage) → string[]`, `saveMetricSelection(storage, keys)`, `featureLabel(name) → string`.

- [ ] **Step 1: Write the failing tests:**

```js
expect(loadMetricSelection(store({}))).toEqual(DEFAULT_METRICS)
expect(loadMetricSelection(store({ 'datect.metrics': 'not json' }))).toEqual(DEFAULT_METRICS)
expect(loadMetricSelection(throwingStore)).toEqual(DEFAULT_METRICS)
expect(loadMetricSelection(store({ 'datect.metrics': '["r2","bogus"]' }))).toEqual(['r2'])
expect(featureLabel('oad_overall_score_30day_mean')).toBe('Ocean anomaly (all regions), 30-day mean')
expect(featureLabel('last_observed_da_raw')).toBe('Last measured DA')
expect(featureLabel('da_raw_prev_obs_2')).toBe('Measurement before last (2 back)')
expect(featureLabel('modis-sst')).toBe('Sea surface temperature (MODIS)')
expect(featureLabel('something_new')).toBe('something_new')
```

- [ ] **Step 2: Run** `npm run test` — Expected: FAIL.
- [ ] **Step 3: Implement** both modules. `featureLabel` is a lookup table for every feature in `final_output.parquet` plus the engine's derived ones (`last_observed_da_raw`, `weeks_since_last_spike`, `da_raw_prev_obs_N`, `month`, OAD lags and windows), with the unknown-name fallback.
- [ ] **Step 4: Run** `npm run test` — Expected: PASS.
- [ ] **Step 5: Implement `MetricsPanel` and the ten tiles** to the mockup's section 6: checkbox row in `METRICS` order, tiles in the same order, each tile naming its window and N. Tile sources: spec section 3 "Metrics, ranked" table; `mae`, `r2` and `line` read `Summary.headline.perSite[site]` (with `headline.overall` for "all sites"), `alert` and `coverage` read `Summary.reliability` (holdout, plus validation for the 2019–21 column); copy for `cats` from `bandVsAlertText`; `prov` shows the forecast's `source` and `modelVersion` and the summary's `source` (the API has no per-summary version stamp, so there is no automatic mismatch flag). Range-dependent tiles (`coverage` stays; `cats`, `width`) render "No range for this forecast" when `band === null`. On a failed forecast (`error` set) `parts` and `drivers` render "No forecast for this week"; the track-record tiles still show.
- [ ] **Step 6: Verify in the preview**: toggling a checkbox persists across reload; Cannon Beach tiles show "Too few 2022–2023 measurements (2)". Screenshot.
- [ ] **Step 7: Commit** `feat(frontend): ranked, selectable metrics panel`.

### Task 9: All-sites table

**Files:**
- Create: `src/components/forecast/AllSitesTable.jsx`
- Modify: `src/pages/Forecast.jsx`

- [ ] **Step 1: Implement** to the mockup's section 7: one `getAllSites(week)` call with its own loading state ("Loading all sites; this can take a minute when forecasts are not precomputed"), started after the site's cards so it never holds them up; a row whose `error` is set shows that error and its last measurement; rows in `config.sites` order with last measured value and "(<age> before)", forecast "[lo–hi]", `<LevelTag>` and probability, range held (`formatPct` or "too few"); `*` after stale ages, `†` after weak ranges, and the two footnotes under the table. Clicking a row sets the page's site. The selected site's row is `bg-gray-50 font-medium`.
- [ ] **Step 2: Verify in the preview**: ten rows north to south; clicking Gold Beach updates every card; the table follows a past week. Screenshot.
- [ ] **Step 3: Commit** `feat(frontend): all-sites table`.

### Task 10: Track record page

**Files:**
- Create: `src/lib/csv.js` (the logic of `exportResultsCSV`, now `exportCsv(rows, filename)`; leave `utils/retrospective.js` alone, Task 12 deletes it), `src/components/trackrecord/{WindowSelector,PerSiteTable,PredictedVsActualChart,ModelComparison}.jsx`
- Test: `src/lib/csv.test.js`
- Modify: `src/pages/TrackRecord.jsx`

**Interfaces:**
- Consumes: `getTrackRecord(windowName, site?)`, `TrackRecord`, `getSiteHistory` and `buildGroundTraces` (Task 7) for the single-site view, `siteReliability`, `formatValue`.
- Produces: `toCsv(rows, columns) → string` (pure; `exportCsv` wraps it with the download).

- [ ] **Step 1: Write the failing test:**

```js
expect(toCsv([{ site: 'Coos Bay', note: 'a "b", c' }], ['site', 'note'])).toBe('site,note\nCoos Bay,"a ""b"", c"')
expect(toCsv([{ site: 'X', v: null }], ['site', 'v'])).toBe('site,v\nX,')
```

- [ ] **Step 2: Run** `npm run test` — Expected: FAIL. **Step 3: Implement** `csv.js`. **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Implement the page.** Window selector (holdout default, validation), site filter (All sites or one). Overall line: N, R² [CI], log R², MAE [CI], range held, onsets caught of N, alarms followed by a jump: all from `TrackRecord.overall`. `PerSiteTable` columns N, R², log R², MAE, range held, onsets caught/N, alarms followed. `PredictedVsActualChart`: all sites = scatter of actual vs ensemble with each point's range as a vertical error bar, the 20 line on both axes; one site = `buildGroundTraces` over that site's history rows inside the window. `ModelComparison`: MAE and R² for ensemble, last value carried forward and Ridge, from the API's summaries. CSV export of the filtered rows. No metric is computed in the browser.
- [ ] **Step 6: Verify in the preview** that the holdout overall line shows R² 0.516 and N = 409 (`docs/CORRECTED_NUMBERS.md`) and Newport's row shows range held 48%. Screenshot.
- [ ] **Step 7: Commit** `feat(frontend): track record page from API summaries`.

### Task 11: Data explorer and About

**Files:**
- Rename: `src/pages/Historical.jsx` → `src/pages/DataExplorer.jsx`
- Modify: `src/pages/DataExplorer.jsx`, `src/pages/About.jsx`, `src/App.jsx` (route import)

- [ ] **Step 1: Restyle `DataExplorer`** per Global Constraints: chart-type buttons become a segmented control of plain text; `react-select` becomes native `<select>`; the waterfall and DA-vs-PN figures already carry the API's subtitle "Weekly maximum; gaps filled by decay", so add no label of your own. Title "Data explorer".
- [ ] **Step 2: Update `About`**: keep the description and data sources; add "What the dashboard cannot do" with the three limits from spec section 4: the range is calibrated over all sites, not per site; `trackRecordText` for the alert's misses and false alarms; data ends <newestMeasurement>. Numbers come from `useConfig` / `getSummary`, not literals.
- [ ] **Step 3: Verify** `npm run lint && npm run test && npm run build`; both pages render in the preview at desktop and 375 px. Screenshot.
- [ ] **Step 4: Commit** `feat(frontend): data explorer and about page`.

### Task 12: Remove the old dashboard

**Files:**
- Delete: `src/pages/Dashboard.jsx`, `src/utils/retrospective.js`, `src/utils/constants.js`
- Modify: `frontend/package.json` (remove `react-datepicker`, `react-select`), `frontend/package-lock.json`; `docs/VISUALIZATIONS_GUIDE.md` and `README.md` lines that describe the old Forecast page (docs only)
- Delete: `docs/superpowers/specs/dashboard-mockup/` except `screenshots/` (the spec still embeds them), and the `dashboard-mockup` entry in `.claude/launch.json` if present

- [ ] **Step 1: Delete and uninstall.** `npm uninstall react-datepicker react-select`.
- [ ] **Step 2: Verify** `grep -rn --exclude-dir=__fixtures__ "retrospective\|Dashboard\|react-select\|react-datepicker\|🎯\|📊\|⚠️\|✅\|💡\|🟢" frontend/src` returns nothing (reword the `retrospective` comment in `services/api.js`); `npm run lint && npm run test && npm run build` pass; `.venv/bin/python -m pytest tests/ -q` still passes (no backend change expected). Compare the built bundle size with `main` (`ls -l dist/assets`) and note it in the PR.
- [ ] **Step 3: Full walk-through in the preview** (desktop and 375 px): Forecast for Newport, Gold Beach, Cannon Beach and a past week with "Show what happened"; Track record holdout and validation; Data explorer; About. Read the console for errors.
- [ ] **Step 4: Commit** `refactor(frontend): remove the old dashboard and unused dependencies`, and update the spec's header (the mockup folder is gone; only its screenshots remain). The controller pushes and opens the PR against `main` after the final review, with before/after screenshots, the bundle sizes, and the list of API fields the frontend now depends on.
