"""Buat dataset model CRP (crp_ifls5_model.csv) dari ifls5_analytic.csv.

Kriteria: ikut subsampel DBS, umur >=15, tidak hamil, hs-CRP <=10 mg/L,
dan prediktor inti lengkap (n akhir = 5.980).
Pemakaian: python src/make_model_dataset.py data/ifls5_analytic.csv data/crp_ifls5_model.csv
"""
import sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)
CORE = ['age', 'female', 'educ', 'pce', 'smoke_status', 'bmi', 'sbp', 'pa_walk_days', 'n_chronic']
c = d[(d.in_dbs == 1) & (d.pregnant != 1) & (d.crp_acute == 0)].dropna(subset=CORE)
COLS = ['pidlink', 'hhid14', 'pid14', 'province', 'crp_plas_equi', 'crp_high', 'pwt14dbsxa',
        'age', 'female', 'married', 'widowed', 'urban', 'educ', 'pce', 'lnpce', 'pce_quintile', 'hhsize',
        'smoke_status', 'cig_day', 'pa_vig_days', 'pa_mod_days', 'pa_walk_days',
        'bmi', 'waist', 'sbp', 'dbp', 'hb',
        'dx_hypert', 'dx_diab', 'dx_heart', 'dx_stroke', 'dx_chol', 'dx_asthma', 'dx_lung', 'dx_arthritis',
        'dx_kidney', 'dx_liver', 'dx_digest', 'n_chronic', 'med_hypert', 'med_diab', 'srh_poor', 'cesd10']
out_df = c[COLS].copy()
out_df['crp_high'] = out_df.crp_high.astype(int)
out_df.to_csv(out, index=False)
print(f'Tersimpan {out}: {len(out_df)} responden, {out_df.crp_high.sum()} CRP tinggi ({out_df.crp_high.mean()*100:.1f}%)')
