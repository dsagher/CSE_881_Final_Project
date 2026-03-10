#!/usr/bin/env python3
"""
Hospital Bed Availability ETL
==============================
Sources (no API key required):
  - HHS HealthData.gov — facility-level weekly hospital capacity  (anag-cw7u)
  - HHS HealthData.gov — state-level daily timeseries             (g62h-syeh)

Both datasets use the Socrata Open Data API (SODA) and are completely public.
Data runs from ~Jan 2020 through May 2024 (collection ended May 3, 2024).

Usage:
    pip install -r requirements.txt
    python etl/pull_hospital_data.py

Output files (data/raw/):
    hhs_facility_timeseries.csv   ~GB-scale, 129 cols, facility × week rows
    hhs_state_timeseries.csv      ~81k rows, 50 cols, state × day rows

Model → field mapping:
  Survival analysis   →  inpatient_beds*, inpatient_beds_used* (derive available = capacity - used)
  Regression / Lasso  →  inpatient_beds_used_covid*, staffed_adult_icu*, admission counts, utilization ratios
  LSTM                →  all of the above as ordered timeseries per facility or state
  Risk score          →  inpatient_bed_covid_utilization*, icu_utilization_*
"""

import csv
import logging
import sys
import time
from io import StringIO
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Socrata settings
PAGE_SIZE   = 50_000   # rows per request (Socrata max is typically 50k)
MAX_RETRIES = 6
SLEEP_BETWEEN_PAGES = 0.3  # seconds — stays well under anonymous rate limits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("etl")


# ── Dataset definitions ───────────────────────────────────────────────────────

DATASETS = [
    {
        "label":    "HHS State Timeseries",
        "url":      "https://healthdata.gov/resource/g62h-syeh.csv",
        "filename": "hhs_state_timeseries.csv",
        "order":    "date,state",
        # ~81k rows × 50 cols — small; downloads in a handful of pages.
        # Key columns for models:
        #   date, state
        #   inpatient_beds, inpatient_beds_used, inpatient_beds_used_covid
        #   staffed_adult_icu_bed_occupancy
        #   total_adult_patients_hospitalized_confirmed_covid
        #   total_pediatric_patients_hospitalized_confirmed_covid
        #   previous_day_admission_adult_covid_confirmed
        #   adult_icu_bed_covid_utilization
        #   inpatient_bed_covid_utilization
        #   critical_staffing_shortage_today_yes, _no, _not_reported
    },
    {
        "label":    "HHS Facility Timeseries",
        "url":      "https://healthdata.gov/resource/anag-cw7u.csv",
        "filename": "hhs_facility_timeseries.csv",
        "order":    "collection_week,hospital_pk",
        # Large dataset: ~129 cols, facility × week.
        # Key columns for models:
        #   hospital_pk, hospital_name, address, city, state, zip, fips_code
        #   collection_week
        #   inpatient_beds_7_day_avg, inpatient_beds_7_day_sum, inpatient_beds_7_day_coverage
        #   inpatient_beds_used_7_day_avg, inpatient_beds_used_7_day_sum
        #   inpatient_beds_used_covid_7_day_avg, inpatient_beds_used_covid_7_day_sum
        #   inpatient_bed_covid_utilization_7_day_avg
        #   staffed_adult_icu_bed_occupancy_7_day_avg
        #   staffed_icu_adult_patients_confirmed_covid_7_day_avg
        #   total_adult_patients_hospitalized_confirmed_covid_7_day_avg
        #   previous_day_admission_adult_covid_confirmed_7_day_sum
        #   previous_day_admission_adult_covid_confirmed_18_49_7_day_sum
        #   previous_day_admission_adult_covid_confirmed_50_69_7_day_sum
        #   previous_day_admission_adult_covid_confirmed_70plus_7_day_sum
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n //= 1024
    return f"{n:.1f} TB"


def _get_with_retry(url: str) -> requests.Response:
    """GET with exponential backoff on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 429:
                wait = 2 ** attempt
                log.warning("  429 rate-limited — sleeping %ds then retry %d/%d", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = 2 ** attempt
            log.warning("  attempt %d/%d failed (%s) — retry in %ds", attempt, MAX_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError("unreachable")  # mypy


def socrata_download(label: str, base_url: str, dest: Path, order: str) -> None:
    """
    Paginate through a Socrata dataset using $limit / $offset and write
    all pages to a single CSV.  Skips the header on pages 2+.
    """
    if dest.exists() and dest.stat().st_size > 0:
        log.info("SKIP  %-40s already on disk (%s)", dest.name, _human(dest.stat().st_size))
        return

    log.info("START %-40s %s", label, dest.name)
    tmp = dest.with_suffix(".tmp")

    offset      = 0
    total_rows  = 0
    first_page  = True

    with open(tmp, "w", newline="", encoding="utf-8") as fout:
        while True:
            url = (
                f"{base_url}"
                f"?$limit={PAGE_SIZE}"
                f"&$offset={offset}"
                f"&$order={order}"
            )
            r = _get_with_retry(url)

            # Parse CSV text to count rows accurately
            lines = r.text.splitlines(keepends=True)
            if len(lines) <= 1:
                log.info("  empty page at offset=%d — done", offset)
                break

            if first_page:
                fout.writelines(lines)
                first_page = False
            else:
                # Skip repeated header on subsequent pages
                fout.writelines(lines[1:])

            rows_this_page = len(lines) - 1   # exclude header
            total_rows    += rows_this_page
            log.info(
                "  page offset=%-7d  rows=%-6d  cumulative=%d",
                offset, rows_this_page, total_rows,
            )

            if rows_this_page < PAGE_SIZE:
                break   # last page — fewer rows than page size means we're done

            offset += PAGE_SIZE
            time.sleep(SLEEP_BETWEEN_PAGES)

    tmp.rename(dest)
    log.info(
        "DONE  %-40s %d rows  %s\n",
        dest.name, total_rows, _human(dest.stat().st_size),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Output directory: %s", OUTPUT_DIR.resolve())
    log.info("Datasets to pull: %d\n", len(DATASETS))

    failures: list[str] = []

    for ds in DATASETS:
        dest = OUTPUT_DIR / ds["filename"]
        try:
            socrata_download(
                label    = ds["label"],
                base_url = ds["url"],
                dest     = dest,
                order    = ds["order"],
            )
        except Exception as exc:
            log.error("FAILED  %s: %s", ds["label"], exc)
            failures.append(ds["label"])

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "── Summary " + "─" * 49)
    for ds in DATASETS:
        p = OUTPUT_DIR / ds["filename"]
        if p.exists():
            size = _human(p.stat().st_size)
            # Quick row count from file
            with open(p, "r", encoding="utf-8") as f:
                rows = sum(1 for _ in f) - 1   # minus header
            print(f"  {ds['filename']:<42}  {rows:>8,} rows   {size}")
        else:
            print(f"  {ds['filename']:<42}  FAILED")
    print("─" * 60 + "\n")

    if failures:
        log.warning("Failed datasets: %s", failures)
        sys.exit(1)
    else:
        log.info("All done. Proceed to preprocessing.")


if __name__ == "__main__":
    main()
