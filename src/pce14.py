"""Pengeluaran per kapita (PCE) IFLS-5 2014.
Port logika do-file resmi RAND/SurveyMETER untuk IFLS East 2012 (pce_ifls_east_2012.do)
ke struktur Buku 1 KS & Buku 2 KR IFLS-5 (kode item identik).
Imputasi nilai hilang: median komunitas -> kecamatan -> kabupaten.
Outlier: winsorize per item pada persentil 99.9 (do-file East memakai cap manual per item).
"""
import sys, numpy as np, pandas as pd
D = sys.argv[1]; OUT = sys.argv[2]
rd = lambda f, cols=None: pd.read_stata(f"{D}/{f}.dta", columns=cols, convert_categoricals=False)

ht = rd('htrack', ['hhid14', 'commid14', 'sc01_14_14', 'sc02_14_14', 'sc03_14_14', 'hwt14xa'])
ht = ht.dropna(subset=['hhid14']).drop_duplicates('hhid14')
ht['kab'] = ht.sc01_14_14 * 100 + ht.sc02_14_14
ht['kec'] = ht.kab * 1000 + ht.sc03_14_14
geo = ht.set_index('hhid14')[['commid14', 'kec', 'kab']]

def impute(w):
    """w: DataFrame index hhid14, kolom item. Isi NaN dgn median komunitas->kec->kab->nasional."""
    w = w.join(geo, how='left')
    items = [c for c in w.columns if c not in geo.columns]
    for c in items:
        hi = w[c].quantile(0.999)
        w[c] = w[c].clip(upper=hi)
        for g in ['commid14', 'kec', 'kab']:
            w[c] = w[c].fillna(w.groupby(g)[c].transform('median'))
        w[c] = w[c].fillna(w[c].median())
    return w[items]

def long_to_wide(f, typ, vals, xcols):
    d = rd(f)
    for v, x in zip(vals, xcols):
        d.loc[d[x] == 3, v] = 0                      # tidak membeli/mengonsumsi = 0
        d.loc[d[x].isin([5, 6, 7, 8, 9]), v] = np.nan  # tidak tahu/missing
    w = d.pivot_table(index='hhid14', columns=typ, values=vals, aggfunc='last')
    w.columns = [f"{a}{b}" for a, b in w.columns]
    return w

# --- Makanan (mingguan) ---
k1 = impute(long_to_wide('b1_ks1', 'ks1type', ['ks02', 'ks03'], ['ks02x', 'ks03x']))
food_items = ['A','AA','B','BA','C','CA','D','DA','E','EA','F','FA','G','GA','H','HA','I','IA','IB',
              'J','K','L','M','N','OA','OB','P','Q','R','S','T','U','V','W','X','Y','Z']
xfood = sum(k1.get(f'ks02{i}', 0) + k1.get(f'ks03{i}', 0) for i in food_items) * 52 / 12
xalc_tob = sum(k1.get(f'ks02{i}', 0) + k1.get(f'ks03{i}', 0) for i in ['FA','GA','HA']) * 52 / 12

# --- Non-makanan bulanan (KS06), tanpa arisan F2 & transfer G masuk total terpisah ---
k2 = long_to_wide('b1_ks2', 'ks2type', ['ks06'], ['ks06x'])
k2 = k2[[c for c in k2.columns if c != 'ks06']]
k2 = impute(k2)
k2['ks06A'] = k2[['ks06A1', 'ks06A2', 'ks06A3', 'ks06A4']].sum(axis=1)
xnonfood2 = k2[['ks06A', 'ks06B', 'ks06C', 'ks06C1', 'ks06D', 'ks06E', 'ks06F1']].sum(axis=1)

# --- Non-makanan tahunan (KS08 beli, KS09a produksi sendiri) ---
k3 = impute(long_to_wide('b1_ks3', 'ks3type', ['ks08', 'ks09a'], ['ks08x', 'ks09ax']))
xnonfood3 = (k3[[f'ks08{i}' for i in 'ABCDEF']].sum(axis=1)
             + k3[[f'ks09a{i}' for i in 'ABCDF' if f'ks09a{i}' in k3]].sum(axis=1)) / 12
xmedical = (k3['ks08C'] + k3.get('ks09aC', 0)) / 12

# --- KS0: non-makanan produksi sendiri (bulanan) & pendidikan anak di rumah (tahunan) ---
k0 = rd('b1_ks0').drop_duplicates('hhid14').set_index('hhid14')
for v in ['ks07a', 'ks10aa', 'ks11aa', 'ks12aa']:
    k0.loc[k0[v + 'x'] == 3, v] = 0
    k0.loc[k0[v + 'x'].isin([5, 6, 7, 8, 9]), v] = np.nan
k0 = impute(k0[['ks07a', 'ks10aa', 'ks11aa', 'ks12aa']])
inonfood = k0['ks07a']
xeduc = (k0['ks10aa'] + k0['ks11aa'] + k0['ks12aa']) / 12

# --- Perumahan (KR04 sewa / KR05 perkiraan sewa rumah sendiri) ---
kr = rd('b2_kr', ['hhid14', 'kr04ax', 'kr05ax', 'kr04a', 'kr05a']).drop_duplicates('hhid14').set_index('hhid14')
for a in ['kr04', 'kr05']:
    v = kr[a + 'a'].where(kr[a + 'a'] < 99999995)
    kr[a] = np.where(kr[a + 'ax'] == 1, v / 12, np.where(kr[a + 'ax'] == 2, v, np.nan))
    kr.loc[kr[a + 'ax'].isna(), a] = np.nan
renter = kr.kr04ax.notna()
kri = impute(kr[['kr04', 'kr05']].assign(kr04=lambda x: x.kr04.where(renter), kr05=lambda x: x.kr05.where(~renter)))
xhouse = np.where(renter, kri.kr04, kri.kr05)
xhouse = pd.Series(xhouse, index=kri.index)

# --- Total ---
df = pd.DataFrame({'xfood': xfood, 'xalc_tob': xalc_tob, 'xnonfood2': xnonfood2, 'xnonfood3': xnonfood3,
                   'xmedical': xmedical, 'inonfood': inonfood, 'xeduc': xeduc, 'xhouse': xhouse})
df['xnonfood'] = df[['xnonfood2', 'xnonfood3', 'inonfood', 'xeduc', 'xhouse']].sum(axis=1, min_count=1)
df['hhexp'] = df.xfood + df.xnonfood

# ukuran RT: anggota yang masih tinggal di RT 2014 (ptrack.member14==1)
pt = rd('ptrack', ['hhid14', 'pid14', 'member14'])
hhsize = pt[pt.member14 == 1].groupby('hhid14').size().rename('hhsize')
df = df.join(hhsize, how='left').join(ht.set_index('hhid14')[['hwt14xa']], how='left')
df['pce'] = df.hhexp / df.hhsize
df = df[df.pce.notna() & (df.pce > 0)]
df['lnpce'] = np.log(df.pce)
df['pce_quintile'] = pd.qcut(df.pce, 5, labels=[1, 2, 3, 4, 5]).astype(int)
df['food_share'] = df.xfood / df.hhexp
df.index.name = 'hhid14'
df.reset_index().to_csv(OUT, index=False)
print(df[['hhexp', 'pce', 'hhsize', 'food_share']].describe(percentiles=[.05, .25, .5, .75, .95]).round(0).to_string())
print('N rumah tangga:', len(df))
