# Benchmark step A: one month zip -> selected columns -> compact dtypes -> Parquet
import sys, zipfile, time, resource, json, os
import pandas as pd
src, y, m, outdir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
t0 = time.time()
use = ["FlightDate","Reporting_Airline","Flight_Number_Reporting_Airline","Origin","Dest","CRSDepTime","CRSArrTime","CRSElapsedTime","Distance","ArrDel15","Cancelled","Diverted","DivReachedDest"]
dt = {"Reporting_Airline":"category","Origin":"category","Dest":"category","Flight_Number_Reporting_Airline":"Int32","CRSDepTime":"Int16","CRSArrTime":"Int16","CRSElapsedTime":"float32","Distance":"float32","ArrDel15":"float32","Cancelled":"float32","Diverted":"float32","DivReachedDest":"float32"}
z = zipfile.ZipFile(f"{src}/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip")
csvn = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
df = pd.read_csv(z.open(csvn), usecols=use, dtype=dt, parse_dates=["FlightDate"])
mem = df.memory_usage(deep=True).sum()/1e6
out = f"{outdir}/{y}_{int(m):02d}.parquet"
df.to_parquet(out, index=False)
lab = df.ArrDel15
res = {"ym": f"{y}-{int(m):02d}", "rows": len(df), "sec": round(time.time()-t0,1),
       "peak_rss_MB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),
       "df_mem_MB": round(mem,1), "parquet_MB": round(os.path.getsize(out)/1e6,1),
       "label_values": sorted(lab.dropna().unique().tolist()),
       "label_missing": int(lab.isna().sum()),
       "cancelled": int((df.Cancelled==1).sum()), "diverted": int((df.Diverted==1).sum()),
       "diverted_with_label": int(((df.Diverted==1)&lab.notna()).sum()),
       "cancelled_with_label": int(((df.Cancelled==1)&lab.notna()).sum()),
       "missing_label_not_canc_or_div": int((lab.isna()&(df.Cancelled!=1)&(df.Diverted!=1)).sum()),
       "diverted_reached_dest": int((df.DivReachedDest==1).sum())}
print(json.dumps(res))
