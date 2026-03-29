"""
Legacy single-row reassignment (stateless). Prefer apply_automation_to_dataframe()
for load-aware, multi-step prevention in the main pipeline.
"""

import pandas as pd

from utils.automation import EmployeePool


def reassign(row: pd.Series):
    if int(row.get("predicted_breach", 0)) != 1:
        return row["assigned_to"]
    pool = EmployeePool()
    emp, _ = pool.pick_reassignee(str(row.get("assigned_to", "")))
    return emp if emp is not None else row["assigned_to"]
