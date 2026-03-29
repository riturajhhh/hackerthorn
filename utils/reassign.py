import pandas as pd

employee_pool = pd.DataFrame({
    "employee": ["A","B","C","D"],
    "skill": [5,4,3,5],
    "load": [3,6,8,2]
})

def reassign(row):
    if row['predicted_breach'] == 1:
        emp = employee_pool.sort_values(['skill','load'], ascending=[False,True]).iloc[0]
        return emp['employee']
    return row['assigned_to']