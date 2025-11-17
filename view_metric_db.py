METRICS_DB_PATH = "/mnt/nfs/mlflow/metrics.db"
import sqlite3
import pandas as pd
import json

def load_into_df():
    conn = sqlite3.connect(METRICS_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT metrics FROM metrics")

    inf_df = pd.DataFrame()
    rows = cur.fetchall()
    

    for (raw,) in rows:
        
        record = json.loads(raw)           # first decode
        if isinstance(record, str):        # second decode if it's still a string
            
            record = json.loads(record)
            run_id = pd.DataFrame([{"mlflow_run_id": record['mlflow_run_id']}], index=[0])
            try:
                fake = (record["inference_metrics"])
            except:
                cur_inf = pd.DataFrame([{"mlflow_run_id": run_id, **record["inf_analysis"]}], index=[0])
    
                inf_df = pd.concat([inf_df, cur_inf], ignore_index=False, axis=0)
    

    inf_df.fillna(float('-inf'), inplace=True)
    conn.close()

    return inf_df

df = load_into_df()

max_mda_values = []
run_ids = []
for i in range(1, 24):
    #print(f"i: {i}")
    best = df.sort_values(by=f"inf_pred_{i}_mda", ascending=False)[:1]
    run_id = best["mlflow_run_id"].values[0]
    run_ids.append(str(run_id['mlflow_run_id'][0]))
    max_mda_values.append(float(best[f"inf_pred_{i}_mda"]))

print("--------------------------------")
print(run_ids)
print("--------------------------------")
print(max_mda_values)

df = pd.concat((pd.Series(run_ids), pd.Series(max_mda_values)), axis=1)
df.columns = ["mlflow_run_id", "max_mda_value"]
df.index = range(1, 24)

print(df)