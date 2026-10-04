"""Trace the A4 cut-faithfulness number (-1.17 at decision step 2500).
Run from inside a folder holding constraint_table.csv plus the weight_reports_*_tradeoff_cut.csv
and recovered_*.csv files for one tradeoff run:  python3 trace_cut_faithfulness.py
Score used by the pipeline for the cut block: 1 - mean absolute error (percentage points) / 100.
"""
import pandas as pd, numpy as np
ct=pd.read_csv('constraint_table.csv')
H=ct.pivot(index='scenario',columns='cut_column',values='cut_percent')[[f'cut{i}' for i in range(1,6)]]
lo=ct.pivot(index='scenario',columns='cut_column',values='range_min')[H.columns]
hi=ct.pivot(index='scenario',columns='cut_column',values='range_max')[H.columns]
cols=[f'report_cut{i}' for i in range(1,6)]
score=lambda R,B: 1-np.mean(np.abs(R-B))/100
def run(tag):
    r=pd.read_csv(f'weight_reports_{tag}_tradeoff_cut.csv'); rec=pd.read_csv(f'recovered_{tag}.csv').set_index('scenario')
    A=r[[f'A_attribute_{i}' for i in range(1,6)]].values; Bo=r[[f'B_attribute_{i}' for i in range(1,6)]].values
    copy=np.all(np.isclose(r[cols].values,A),1)|np.all(np.isclose(r[cols].values,Bo),1)
    s=r.scenario.values
    pct=(r[cols].values-lo.loc[s].values)/(hi.loc[s].values-lo.loc[s].values)*100
    rp=r.copy(); rp[cols]=np.clip(pct,0,100)
    out={}
    def sc(df,target):
        R=df.groupby('scenario')[cols].mean(); idx=R.index
        T=(rec.loc[idx,[f'b_cut{i}' for i in range(1,6)]].values if target=='rec' else H.loc[idx].values)
        return score(R.values,T), len(idx)
    out['as scored now (raw, vs recovered)']=sc(r,'rec')
    out['all-zero report (vs recovered)']=(score(np.zeros((len(rec),5)),rec[[f'b_cut{i}' for i in range(1,6)]].values),len(rec))
    out['converted to % of range (vs recovered)']=sc(rp,'rec')
    out['converted, copies removed (vs recovered)']=sc(rp[~copy],'rec')
    out['converted, copies removed (vs TRUE hidden)']=sc(rp[~copy],'hid')
    out['all-zero report (vs TRUE hidden)']=(score(np.zeros((len(rec),5)),H.loc[rec.index].values),len(rec))
    print(f'== {tag}   copies of a shown option: {copy.mean():.0%}')
    for k,(v,n) in out.items(): print(f'  {k:45s}{v:7.3f}  (n={n})')
for t in ['decision_step-500','decision_step-2500','introspection-fold1_step-300','introspection-fold2_step-300']: run(t)
