"""Bangun dataset analitik individu IFLS-5 (2014) untuk topik hs-CRP dan lansia."""
import sys, numpy as np, pandas as pd
D, PCE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
K = ['hhid14', 'pid14']
rd = lambda f, cols=None: pd.read_stata(f"{D}/{f}.dta", columns=cols, convert_categoricals=False)
def miss(s, codes=(8, 9, 98, 99, 998, 999)):
    return s.where(~s.isin(codes))

# --- Demografi (roster, anggota 2014) ---
ar = rd('bk_ar1', K + ['pidlink', 'ar07', 'ar09', 'ar13'])
pt = rd('ptrack', K + ['member14', 'pwt14xa'])
base = ar.merge(pt, on=K, how='left')
base = base[(base.member14 == 1) & (base.ar09.between(15, 120))].copy()
base['female'] = (base.ar07 == 3).astype(float).where(base.ar07.notna())
base['age'] = base.ar09
base['married'] = (base.ar13 == 2).astype(float).where(miss(base.ar13).notna())
base['widowed'] = (base.ar13 == 5).astype(float).where(miss(base.ar13).notna())

sc = rd('bk_sc1', ['hhid14', 'sc05', 'sc01_14_14']).drop_duplicates('hhid14')
sc['urban'] = (sc.sc05 == 1).astype(float); sc['province'] = sc.sc01_14_14
base = base.merge(sc[['hhid14', 'urban', 'province']], on='hhid14', how='left')

# --- Pendidikan ---
dl = rd('b3a_dl1', K + ['dl04', 'dl06'])
lvl = {2: 1, 72: 1, 11: 1, 3: 2, 4: 2, 73: 2, 12: 2, 5: 3, 6: 3, 74: 3, 15: 3, 14: 2,
       60: 4, 61: 4, 62: 4, 63: 4, 13: 4, 17: 1, 90: 0}
dl['educ'] = np.where(dl.dl04 == 3, 0, dl.dl06.map(lvl))  # 0 tdk sekolah,1 SD,2 SMP,3 SMA,4 PT
base = base.merge(dl[K + ['educ']], on=K, how='left')

# --- SES: PCE rumah tangga ---
pce = pd.read_csv(PCE, usecols=['hhid14', 'pce', 'lnpce', 'pce_quintile', 'hhsize'])
base = base.merge(pce, on='hhid14', how='left')

# --- Merokok ---
km = rd('b3b_km', K + ['km01a', 'km01e', 'km04', 'km08'])
km['ever_smoke'] = (km.km01a == 1).astype(float).where(km.km01a.isin([1, 3]))
km['smoke_status'] = np.select([km.km01a == 3, (km.km01a == 1) & (km.km04 == 3), (km.km01a == 1) & (km.km04 == 1)],
                               [0, 1, 2], np.nan)  # 0 tdk pernah,1 mantan,2 aktif
km['cig_day'] = np.where(km.smoke_status == 2, km.km08, 0); km.loc[km.smoke_status.isna(), 'cig_day'] = np.nan
base = base.merge(km[K + ['smoke_status', 'cig_day']], on=K, how='left')

# --- Aktivitas fisik (hari/minggu: A berat, B sedang, C jalan kaki) ---
kk2 = rd('b3b_kk2', K + ['kktype', 'kk02m', 'kk02o'])
kk2['days'] = np.where(kk2.kk02m == 3, 0, miss(kk2.kk02o, (8, 9, 98, 99)))
pa = kk2.pivot_table(index=K, columns='kktype', values='days', aggfunc='last')
pa.columns = ['pa_vig_days', 'pa_mod_days', 'pa_walk_days']
base = base.merge(pa.reset_index(), on=K, how='left')

# --- Kesehatan subjektif ---
kk1 = rd('b3b_kk1', K + ['kk01'])
kk1['srh_poor'] = kk1.kk01.isin([3, 4]).astype(float).where(kk1.kk01.isin([1, 2, 3, 4]))
base = base.merge(kk1[K + ['srh_poor']], on=K, how='left')

# --- Penyakit kronis terdiagnosis (CD05) ---
cdmap = {'A': 'dx_hypert', 'B': 'dx_diab', 'C': 'dx_tb', 'D': 'dx_asthma', 'E': 'dx_lung', 'F': 'dx_heart',
         'G': 'dx_liver', 'H': 'dx_stroke', 'I': 'dx_cancer', 'J': 'dx_arthritis', 'M': 'dx_chol',
         'N': 'dx_prostate', 'O': 'dx_kidney', 'P': 'dx_digest', 'Q': 'dx_psych', 'R': 'dx_memory'}
cd = rd('b3b_cd3', K + ['cdtype', 'cd05'])
cd['y'] = (cd.cd05 == 1).astype(float).where(cd.cd05.isin([1, 3]))
cdw = cd.pivot_table(index=K, columns='cdtype', values='y', aggfunc='last').rename(columns=cdmap)
cdw['n_chronic'] = cdw[list(cdmap.values())].sum(axis=1, min_count=8)
base = base.merge(cdw.reset_index(), on=K, how='left')

# --- CES-D 10 (item E 'hopeful' dan H 'happy' dibalik) ---
kp = rd('b3b_kp', K + ['kptype', 'kp02'])
kp['v'] = miss(kp.kp02, (9,)) - 1
kpw = kp.pivot_table(index=K, columns='kptype', values='v', aggfunc='last')
for c in ['E', 'H']: kpw[c] = 3 - kpw[c]
kpw['cesd10'] = kpw[list('ABCDEFGHIJ')].sum(axis=1, min_count=10)
kpw['depressed'] = (kpw.cesd10 >= 10).astype(float).where(kpw.cesd10.notna())
base = base.merge(kpw[['cesd10', 'depressed']].reset_index(), on=K, how='left')

# --- Kognisi (ingat kata segera + tertunda) ---
co = rd('b3b_co1', K + ['co07count', 'co10count'])
co['word_recall'] = co.co07count + co.co10count
base = base.merge(co[K + ['word_recall']], on=K, how='left')

# --- ADL/IADL (KK03: 1 mudah, 2 sulit, 3 tdk bisa) ---
kk3 = rd('b3b_kk3', K + ['kk3type', 'kk03'])
kk3['lim'] = kk3.kk03.isin([2, 3]).astype(float).where(kk3.kk03.isin([1, 2, 3]))
adl = kk3.groupby(K).lim.sum(min_count=15).rename('n_func_limit').reset_index()
base = base.merge(adl, on=K, how='left')

# --- Pengukuran fisik (Buku US) ---
us = rd('bus_us')
num = lambda c: us[c].where((us[c] > 0) & (us[c] < 990))
us['height'] = num('us04'); us['weight'] = num('us06')
us['bmi'] = us.weight / (us.height / 100) ** 2
us.loc[~us.bmi.between(10, 70), 'bmi'] = np.nan
us['waist'] = num('us06a')
sbp = pd.concat([num(c) for c in ['us07b1', 'us07c1']], axis=1).mean(axis=1)
dbp = pd.concat([num(c) for c in ['us07b2', 'us07c2']], axis=1).mean(axis=1)
us['sbp'] = sbp.fillna(num('us07a1')); us['dbp'] = dbp.fillna(num('us07a2'))
us['grip_max'] = pd.concat([num(c) for c in ['us20a', 'us20b', 'us20c', 'us20d']], axis=1).max(axis=1)
us['hb'] = us.us13.where(us.us13.between(3, 25))
us['sit_stand_sec'] = num('us10_1')
us['walk_sec'] = pd.concat([num('us19q'), num('us19r')], axis=1).min(axis=1)
for c, n in [('us18ab', 'med_hypert'), ('us18ac', 'med_diab'), ('us18ad', 'med_chol')]:
    us[n] = (us[c] == 1).astype(float).where(us[c].isin([1, 3]))
us['pregnant'] = (us.us15a == 1).astype(float)
base = base.merge(us[K + ['height', 'weight', 'bmi', 'waist', 'sbp', 'dbp', 'grip_max', 'hb', 'sit_stand_sec',
                          'walk_sec', 'med_hypert', 'med_diab', 'med_chol', 'pregnant']], on=K, how='left')
base['hypert_measured'] = ((base.sbp >= 140) | (base.dbp >= 90) | (base.med_hypert == 1)).astype(float).where(base.sbp.notna())

# --- DBS: hs-CRP & HbA1c ---
dbs = rd('dbs_ifls5_public_use', ['hhid14', 'pid14', 'crp_dbs', 'crp_plas_equi', 'lncrp_plas_equi',
                                  'a1c_rev', 'a1c_revx', 'pwt14dbsxa'])
dbs['crp_high'] = (dbs.crp_plas_equi >= 3).astype(float)
dbs['crp_acute'] = (dbs.crp_plas_equi > 10).astype(float)
dbs['in_dbs'] = 1
base = base.merge(dbs, on=K, how='left')
base['in_dbs'] = base.in_dbs.fillna(0)
base['elderly60'] = (base.age >= 60).astype(int)
base.to_csv(OUT, index=False)
print('N dewasa >=15 di roster 2014:', len(base))
