"""
NGAFID-MC aviation dataset vocabulary: the one place that knows the raw
long-format sensor CSV's column names, grouping key, and label values.
Previously duplicated across scripts/featurize_ngafid_flights.py,
orchestrator/dynamic_loop.py's deep_dive branch, and
scripts/run_deep_dive_agent.py — consolidated here once a fourth
consumer (dynamic_loop.py's featurize_timeseries branch) made the
duplication worth removing. Every other module in this pipeline stays
dataset-agnostic; only this file is allowed to know these specifics.
"""
from __future__ import annotations

NGAFID_SENSORS = [
    "volt1", "volt2", "amp1", "amp2", "FQtyL", "FQtyR", "E1 FFlow",
    "E1 OilT", "E1 OilP", "E1 RPM", "E1 CHT1", "E1 CHT2", "E1 CHT3",
    "E1 CHT4", "E1 EGT1", "E1 EGT2", "E1 EGT3", "E1 EGT4", "OAT",
    "IAS", "VSpd", "NormAc", "AltMSL",
]

NGAFID_ID_COLUMN = "id"
NGAFID_GROUP_COLUMN = "plane_id"
NGAFID_LABEL_COLUMN = "before_after"
NGAFID_EXTRA_COLUMNS = ["date_diff", "split"]

# before_after: 1 = flight occurred before the recorded maintenance event
# (the flight the label is trying to flag), 0 = after. Accepts a few
# variant spellings since the raw dataset's exact casing/string form
# isn't guaranteed across exports.
NGAFID_LABEL_MAP = {
    "before": 1, "pre": 1, "pre-maintenance": 1, "1": 1, "true": 1,
    "after": 0, "post": 0, "post-maintenance": 0, "0": 0, "false": 0,
}
