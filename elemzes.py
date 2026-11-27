# -*- coding: utf-8 -*-
"""
Társadalmi és gazdasági tényezők összefüggése az inkumbens párt támogatottságának
változásával a 2022-es és 2026-os magyar választásokon

Besze Zsolt, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.iolib.summary2 import summary_col

# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 1. VÁLASZTÁS 2022
# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

valasztas22 = pd.read_excel("_Választási adatok/2022/A2022.éviországgyűlésiválasztásjogerőseredményei/Listás_szavazás_szkjkv.xlsx")

# Azonosítók: szavazókör (F/T jel nélkül) és település (NVI-kód)
valasztas22["KÖR_ID"] = valasztas22["SZAVAZÓKÖR_AZON"].str[:-1]
valasztas22["TELEP_KÓD"] = valasztas22["KÖR_ID"].str[:5]

# F sorok: szavazóköri összesítők, T sorok: listánkénti szavazatok
fej22 = valasztas22[valasztas22["SZAVAZÓKÖR_AZON"].str.endswith("F")]
tetel22 = valasztas22[valasztas22["SZAVAZÓKÖR_AZON"].str.endswith("T")]

# 1.1 Pártlistás érvényes szavazatok (nemzetiségi listák nélkül)
partlistas22 = tetel22[tetel22["LISTA_TÍPUS"] != "Nemzetiségi"]

partlistas_ossz22 = partlistas22.groupby("KÖR_ID")["SZAVAZAT"].sum().reset_index()
partlistas_ossz22.columns = ["KÖR_ID", "PÁRTLISTÁS_ÉRVÉNYES"]

# 1.2 Fidesz–KDNP szavazatok
fidesz22 = partlistas22[partlistas22["LISTA"] == "FIDESZ - MAGYAR POLGÁRI SZÖVETSÉG-KERESZTÉNYDEMOKRATA NÉPPÁRT"]
fidesz22 = fidesz22[["KÖR_ID", "SZAVAZAT"]]
fidesz22.columns = ["KÖR_ID", "FIDESZ"]

# 1.3 Szavazóköri tábla
korok22 = pd.merge(fej22, partlistas_ossz22, how="left", on="KÖR_ID")
korok22 = pd.merge(korok22, fidesz22, how="left", on="KÖR_ID")
print(korok22[["PÁRTLISTÁS_ÉRVÉNYES", "FIDESZ"]].isnull().sum())   # 0 és 0

# Egyszeri ellenőrzés: összevetés a hivatalos eredménnyel
print(korok22["FIDESZ"].sum())   # 2 809 238 (hivatalos hazai listás szavazat)

# 1.4 Számlálókörök: OEVK-nként a legnagyobb névjegyzékű kör
oevk_max22 = korok22.groupby(["MEGYE", "OEVK"])["VÁLASZTÓPOLGÁR"].max().reset_index()
oevk_max22.columns = ["MEGYE", "OEVK", "OEVK_MAX_VP"]

korok22 = pd.merge(korok22, oevk_max22, how="left", on=["MEGYE", "OEVK"])
korok22["SZÁMLÁLÓKÖR"] = korok22["VÁLASZTÓPOLGÁR"] == korok22["OEVK_MAX_VP"]

print(korok22["SZÁMLÁLÓKÖR"].sum())   # 106

korok22 = korok22[["KÖR_ID", "TELEP_KÓD", "MEGYE", "TELEPÜLÉS", "VÁLASZTÓPOLGÁR",
                   "MEGJELENTEK", "PÁRTLISTÁS_ÉRVÉNYES", "FIDESZ", "SZÁMLÁLÓKÖR"]]

# 1.5 Település/kerület szint, számlálókörök nélkül
korok22_szk_nelkul = korok22[korok22["SZÁMLÁLÓKÖR"] == False]
print(len(korok22) - len(korok22_szk_nelkul))   # 106
print(korok22[korok22["SZÁMLÁLÓKÖR"]]["TELEP_KÓD"].nunique())   # 97

# A kizárt számlálókörök súlya: a pártlistás szavazatok hány százaléka
print(korok22[korok22["SZÁMLÁLÓKÖR"]]["PÁRTLISTÁS_ÉRVÉNYES"].sum() / korok22["PÁRTLISTÁS_ÉRVÉNYES"].sum() * 100)   # 4,7

telep22 = korok22_szk_nelkul.groupby(["TELEP_KÓD", "MEGYE", "TELEPÜLÉS"])[["VÁLASZTÓPOLGÁR", "MEGJELENTEK", "PÁRTLISTÁS_ÉRVÉNYES", "FIDESZ"]].sum().reset_index()
print(len(telep22))   # 3177 (3154 település + 23 kerület)

# 1.6 Fidesz-arány és részvétel (%)
telep22["FIDESZ_ARÁNY22"] = telep22["FIDESZ"] / telep22["PÁRTLISTÁS_ÉRVÉNYES"] * 100
telep22["RÉSZVÉTEL22"] = telep22["MEGJELENTEK"] / telep22["VÁLASZTÓPOLGÁR"] * 100

print(telep22["FIDESZ"].sum() / telep22["PÁRTLISTÁS_ÉRVÉNYES"].sum() * 100)   # 53,25
print(telep22["MEGJELENTEK"].sum() / telep22["VÁLASZTÓPOLGÁR"].sum() * 100)   # 69,54 (országos részvétel, számlálókörök nélkül)


# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 2. VÁLASZTÁS 2026
# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# 2.1 A 20 megyei fájl beolvasása és egymás alá fűzése
megyek = ["Bács-Kiskun", "Baranya", "Békés", "Borsod-Abaúj-Zemplén", "Budapest",
          "Csongrád-Csanád", "Fejér", "Győr-Moson-Sopron", "Hajdú-Bihar", "Heves",
          "Jász-Nagykun-Szolnok", "Komárom-Esztergom", "Nógrád", "Pest", "Somogy",
          "Szabolcs-Szatmár-Bereg", "Tolna", "Vas", "Veszprém", "Zala"]

tablak26 = []
for megye in megyek:
    aktualis_tabla = pd.read_excel("_Választási adatok/2026/Varmegye_Listás_szavazokori_eredmenyek_2026/" + megye + " listás 2026.xls",
                                   sheet_name=1, header=1)
    tablak26.append(aktualis_tabla)

valasztas26 = pd.concat(tablak26, ignore_index=True)

# Azonosítók: a kötőjelek nélküli kód első 8 jegye a szavazókör, első 5 jegye a település
valasztas26["KÖR_ID"] = valasztas26["Szavazókör azonosító"].str.replace("-", "").str[:8]
valasztas26["TELEP_KÓD"] = valasztas26["KÖR_ID"].str[:5]

# 2.2 Pártlistás érvényes szavazatok (01–05: MKKP, TISZA, Mi Hazánk, DK, FIDESZ-KDNP)
valasztas26["PÁRTLISTÁS_ÉRVÉNYES"] = valasztas26["01"] + valasztas26["02"] + valasztas26["03"] + valasztas26["04"] + valasztas26["05"]

# 2.3 Számlálókörök: ahová átjelentkezéssel / külképviseleten leadott borítékok érkeztek
valasztas26["SZÁMLÁLÓKÖR"] = valasztas26["IL"] > 0
print(valasztas26["SZÁMLÁLÓKÖR"].sum())   # 106

# Szavazóköri tábla a 2022-essel egyező oszlopnevekkel
korok26 = valasztas26[["KÖR_ID", "TELEP_KÓD", "Település", "EL", "JL",
                       "PÁRTLISTÁS_ÉRVÉNYES", "05", "SZÁMLÁLÓKÖR"]]
korok26.columns = ["KÖR_ID", "TELEP_KÓD", "TELEPÜLÉS", "VÁLASZTÓPOLGÁR", "MEGJELENTEK",
                   "PÁRTLISTÁS_ÉRVÉNYES", "FIDESZ", "SZÁMLÁLÓKÖR"]

# Egyszeri ellenőrzés: összevetés a hivatalos eredménnyel
print(korok26["FIDESZ"].sum())   # 2 175 671 (hivatalos hazai listás szavazat)

# 2.4 Település/kerület szint, számlálókörök nélkül
korok26_szk_nelkul = korok26[korok26["SZÁMLÁLÓKÖR"] == False]
print(len(korok26) - len(korok26_szk_nelkul))   # 106
print(korok26[korok26["SZÁMLÁLÓKÖR"]]["TELEP_KÓD"].nunique())   # 97

# A kizárt számlálókörök súlya: a pártlistás szavazatok hány százaléka
print(korok26[korok26["SZÁMLÁLÓKÖR"]]["PÁRTLISTÁS_ÉRVÉNYES"].sum() / korok26["PÁRTLISTÁS_ÉRVÉNYES"].sum() * 100)   # 5,9

telep26 = korok26_szk_nelkul.groupby(["TELEP_KÓD", "TELEPÜLÉS"])[["VÁLASZTÓPOLGÁR", "MEGJELENTEK", "PÁRTLISTÁS_ÉRVÉNYES", "FIDESZ"]].sum().reset_index()
print(len(telep26))   # 3177

# 2.5 Fidesz-arány és részvétel (%)
telep26["FIDESZ_ARÁNY26"] = telep26["FIDESZ"] / telep26["PÁRTLISTÁS_ÉRVÉNYES"] * 100
telep26["RÉSZVÉTEL26"] = telep26["MEGJELENTEK"] / telep26["VÁLASZTÓPOLGÁR"] * 100

print(telep26["FIDESZ"].sum() / telep26["PÁRTLISTÁS_ÉRVÉNYES"].sum() * 100)   # 37,24
print(telep26["MEGJELENTEK"].sum() / telep26["VÁLASZTÓPOLGÁR"].sum() * 100)   # 78,99 (országos részvétel, számlálókörök nélkül)


# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 3. A KÉT VÁLASZTÁS ÖSSZEKAPCSOLÁSA, A FIDESZ-VÁLTOZÁS
# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# 3.1 A két év települési táblája évvégződéses oszlopnevekkel
ossze22 = telep22[["TELEP_KÓD", "MEGYE", "TELEPÜLÉS", "PÁRTLISTÁS_ÉRVÉNYES", "FIDESZ_ARÁNY22", "RÉSZVÉTEL22"]]
ossze22.columns = ["TELEP_KÓD", "MEGYE", "TELEPÜLÉS", "ÉRVÉNYES22", "FIDESZ_ARÁNY22", "RÉSZVÉTEL22"]

ossze26 = telep26[["TELEP_KÓD", "TELEPÜLÉS", "PÁRTLISTÁS_ÉRVÉNYES", "FIDESZ_ARÁNY26", "RÉSZVÉTEL26"]]
ossze26.columns = ["TELEP_KÓD", "TELEPÜLÉS26", "ÉRVÉNYES26", "FIDESZ_ARÁNY26", "RÉSZVÉTEL26"]

# 3.2 Összekapcsolás a településkóddal
telep = pd.merge(ossze22, ossze26, how="outer", on="TELEP_KÓD")

print(len(telep))                   # 3177
print(telep.isnull().sum().sum())   # 0: minden kód mindkét évben megvan
print(telep[telep["TELEPÜLÉS"] != telep["TELEPÜLÉS26"]])   # üres: a kód mindkét évben ugyanazt a települést jelöli

# 3.3 Az eredményváltozó: a Fidesz-arány változása (százalékpont)
telep["FIDESZ_VÁLTOZÁS"] = telep["FIDESZ_ARÁNY26"] - telep["FIDESZ_ARÁNY22"]

# Fidesz-arány és részvétel változások iránya a településeken
print((telep["FIDESZ_VÁLTOZÁS"] > 0).sum(), (telep["FIDESZ_VÁLTOZÁS"] == 0).sum())   # 7 nőtt, 1 nem változott
print((telep["RÉSZVÉTEL26"] - telep["RÉSZVÉTEL22"] < 0).sum(), (telep["RÉSZVÉTEL26"] - telep["RÉSZVÉTEL22"] == 0).sum())   # 99 csökkent, 3 nem változott

# Ellenőrzéshez: az összekapcsolt tábla Excelbe
telep.to_excel("Export/ellenorzes_telep.xlsx", index=False)

# Arányos visszaesés esetén várható β: az országos Fidesz-arány relatív változása
print(telep26["FIDESZ"].sum() / telep26["PÁRTLISTÁS_ÉRVÉNYES"].sum() / 
      (telep22["FIDESZ"].sum() / telep22["PÁRTLISTÁS_ÉRVÉNYES"].sum()) - 1)   # −0,30: az arányos visszaesés esetén várható β (4.2)


# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 4. MAGYARÁZÓVÁLTOZÓK
# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# 4.1 Beolvasás: évente és egységtípusonként a fő adatok és a lakásadatok
tstar21 = pd.read_excel("TSTAR/tstar2021_településstatisztikai_adatok.xlsx")
tstar21_lak = pd.read_excel("TSTAR/tstar2021_településstatisztikai_adatok_lakás.xlsx")
bpstar21 = pd.read_excel("BPSTAR/bpstar2021_településstatisztikai_adatok.xlsx")
bpstar21_lak = pd.read_excel("BPSTAR/bpstar2021_településstatisztikai_adatok_lakás.xlsx")

tstar25 = pd.read_excel("TSTAR/tstar2025_településstatisztikai_adatok.xlsx")
tstar25_lak = pd.read_excel("TSTAR/tstar2025_településstatisztikai_adatok_lakás.xlsx")
bpstar25 = pd.read_excel("BPSTAR/bpstar2025_településstatisztikai_adatok.xlsx")
bpstar25_lak = pd.read_excel("BPSTAR/bpstar2025_településstatisztikai_adatok_lakás.xlsx")

print(len(tstar21), len(tstar21_lak), len(bpstar21), len(bpstar21_lak))   # 3154 3154 23 23
print(len(tstar25), len(tstar25_lak), len(bpstar25), len(bpstar25_lak))   # 3154 3154 23 23

# 4.2 Évente egy tábla: a települések és a kerületek egymás alatt, mellettük a lakásadatok
ksh21 = pd.concat([tstar21, bpstar21], ignore_index=True)
ksh21_lak = pd.concat([tstar21_lak, bpstar21_lak], ignore_index=True)
ksh21_lak = ksh21_lak.iloc[:, 0:6]   # Terület + 5 lakásoszlop (a „Települések” nélkül)
ksh21 = pd.merge(ksh21, ksh21_lak, how="left", on="Terület")

ksh25 = pd.concat([tstar25, bpstar25], ignore_index=True)
ksh25_lak = pd.concat([tstar25_lak, bpstar25_lak], ignore_index=True)
ksh25_lak = ksh25_lak.iloc[:, 0:6]
ksh25 = pd.merge(ksh25, ksh25_lak, how="left", on="Terület")

ksh21.columns = ["TELEPÜLÉS", "NÉPESSÉG21", "GYEREK21", "IDŐS21", "AKTÍVKORÚ21",
                 "ÁLLÁSKERESŐ21", "KÖZFOGL21", "FHT21", "RGYK21", "TERÜLET21",
                 "ODAVÁND21", "ELVÁND21", "AUTÓ21", "KSH_TELEPÜLÉS",
                 "LAKÁS21", "LAKÁS_1SZ21", "LAKÁS_2SZ21", "LAKÁS_3SZ21", "LAKÁS_4SZ21"]
ksh25.columns = ["TELEPÜLÉS", "NÉPESSÉG25", "GYEREK25", "IDŐS25", "AKTÍVKORÚ25",
                 "ÁLLÁSKERESŐ25", "KÖZFOGL25", "FHT25", "RGYK25", "TERÜLET25",
                 "ODAVÁND25", "ELVÁND25", "AUTÓ25", "KSH_TELEPÜLÉS",
                 "LAKÁS25", "LAKÁS_1SZ25", "LAKÁS_2SZ25", "LAKÁS_3SZ25", "LAKÁS_4SZ25"]

print(ksh21["LAKÁS21"].isnull().sum(), ksh25["LAKÁS25"].isnull().sum())   # 0 0: minden sorhoz került lakásadat

# 4.3 Üres cellák = 0 (a Statinfo a nullát üresen exportálja)
# A számoszlopok: 1–12. és 14–18. (a 0. a név, a 13. a KSH_TELEPÜLÉS, az maradjon üres a kerületeknél)
ksh21.iloc[:, 1:13] = ksh21.iloc[:, 1:13].fillna(0)
ksh21.iloc[:, 14:19] = ksh21.iloc[:, 14:19].fillna(0)
ksh25.iloc[:, 1:13] = ksh25.iloc[:, 1:13].fillna(0)
ksh25.iloc[:, 14:19] = ksh25.iloc[:, 14:19].fillna(0)

print(ksh21.isnull().sum().sum(), ksh25.isnull().sum().sum())   # 23 23 (csak a kerületek KSH-kódja)
print((ksh21["GYEREK21"] + ksh21["IDŐS21"] + ksh21["AKTÍVKORÚ21"] != ksh21["NÉPESSÉG21"]).sum())   # 0
print((ksh25["LAKÁS_1SZ25"] + ksh25["LAKÁS_2SZ25"] + ksh25["LAKÁS_3SZ25"] + ksh25["LAKÁS_4SZ25"] != ksh25["LAKÁS25"]).sum())   # 0

# 4.4 Kerületnevek
kerulet_nevek = {"Budapest 01. ker.": "Budapest I. kerület", "Budapest 02. ker.": "Budapest II. kerület",
                 "Budapest 03. ker.": "Budapest III. kerület", "Budapest 04. ker.": "Budapest IV. kerület",
                 "Budapest 05. ker.": "Budapest V. kerület", "Budapest 06. ker.": "Budapest VI. kerület",
                 "Budapest 07. ker.": "Budapest VII. kerület", "Budapest 08. ker.": "Budapest VIII. kerület",
                 "Budapest 09. ker.": "Budapest IX. kerület", "Budapest 10. ker.": "Budapest X. kerület",
                 "Budapest 11. ker.": "Budapest XI. kerület", "Budapest 12. ker.": "Budapest XII. kerület",
                 "Budapest 13. ker.": "Budapest XIII. kerület", "Budapest 14. ker.": "Budapest XIV. kerület",
                 "Budapest 15. ker.": "Budapest XV. kerület", "Budapest 16. ker.": "Budapest XVI. kerület",
                 "Budapest 17. ker.": "Budapest XVII. kerület", "Budapest 18. ker.": "Budapest XVIII. kerület",
                 "Budapest 19. ker.": "Budapest XIX. kerület", "Budapest 20. ker.": "Budapest XX. kerület",
                 "Budapest 21. ker.": "Budapest XXI. kerület", "Budapest 22. ker.": "Budapest XXII. kerület",
                 "Budapest 23. ker.": "Budapest XXIII. kerület"}

ksh21["TELEPÜLÉS"] = ksh21["TELEPÜLÉS"].replace(kerulet_nevek)
ksh25["TELEPÜLÉS"] = ksh25["TELEPÜLÉS"].replace(kerulet_nevek)

# Budapest súlya: lakónépesség és arány az ország népességében (3.7)
bp21 = ksh21[ksh21["TELEPÜLÉS"].str.startswith("Budapest")]["NÉPESSÉG21"].sum()
bp25 = ksh25[ksh25["TELEPÜLÉS"].str.startswith("Budapest")]["NÉPESSÉG25"].sum()
print(bp21, bp21 / ksh21["NÉPESSÉG21"].sum() * 100)   # 1 672 443 és 17,40
print(bp25, bp25 / ksh25["NÉPESSÉG25"].sum() * 100)   # 1 675 777 és 17,66

print(ksh21["TELEPÜLÉS"].isin(telep["TELEPÜLÉS"]).sum(), ksh25["TELEPÜLÉS"].isin(telep["TELEPÜLÉS"]).sum())   # 3177 3177

# 4.5 A népsűrűség eloszlása: nyers vagy logaritmus? (1. ábra)
nepsuruseg21 = ksh21["NÉPESSÉG21"] / ksh21["TERÜLET21"]

# Ferdeség a transzformáció előtt és után
print(nepsuruseg21.skew(), np.log(nepsuruseg21).skew())   # 23,10 és 0,91

# Osztályközök száma: a legkisebb k, amelyre 2^k >= N
k = int(np.ceil(np.log2(len(nepsuruseg21))))
print(k)   # 12

# Ezres tagolás (pl. 20 000)
def ezres(ertek, pozicio):
    return f"{ertek:,.0f}".replace(",", " ")

# Adatfeliratok az oszlopok fölé (az üres osztályközök fölé nem)
def feliratok(darab):
    return [ezres(d, None) if d > 0 else "" for d in darab]

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
szin = "#3B6EA8"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 5.0))

# a) nyers értékek
darab, hatarok, oszlopok = ax1.hist(nepsuruseg21, bins=k, color=szin, edgecolor="white")
ax1.bar_label(oszlopok, labels=feliratok(darab), fontsize=10, padding=1)
ax1.set_title("a) Nyers értékek", pad=16)
ax1.set_xlabel("Népsűrűség (fő/km²)")
ax1.xaxis.set_major_formatter(FuncFormatter(ezres))

# b) logaritmus
darab, hatarok, oszlopok = ax2.hist(np.log(nepsuruseg21), bins=k, color=szin, edgecolor="white")
ax2.bar_label(oszlopok, labels=feliratok(darab), fontsize=10, padding=1)
ax2.set_title("b) Természetes alapú logaritmus", pad=16)
ax2.set_xlabel("ln(népsűrűség)\n(2 ≈ 7,  4 ≈ 55,  6 ≈ 400,  8 ≈ 3000 fő/km²)")
ax2.set_xticks([0, 2, 4, 6, 8, 10])

# közös kinézet
for ax in [ax1, ax2]:
    ax.set_ylabel("Települések / kerületek száma")
    ax.yaxis.set_major_formatter(FuncFormatter(ezres))
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout(h_pad=4)
plt.savefig("ÁBRÁK/nepsuruseg.png", dpi=300, bbox_inches="tight")
plt.show()

# 4.6 Arányok: korösszetétel, munkaerőpiac, autó, vándorlás, népsűrűség
ksh21["GYEREK_ARÁNY21"] = ksh21["GYEREK21"] / ksh21["NÉPESSÉG21"] * 100
ksh21["IDŐS_ARÁNY21"] = ksh21["IDŐS21"] / ksh21["NÉPESSÉG21"] * 100
ksh21["ÁLLÁSKERESŐ_ARÁNY21"] = ksh21["ÁLLÁSKERESŐ21"] / ksh21["AKTÍVKORÚ21"] * 100
ksh21["KÖZFOGL_ARÁNY21"] = ksh21["KÖZFOGL21"] / ksh21["AKTÍVKORÚ21"] * 100
ksh21["FHT_ARÁNY21"] = ksh21["FHT21"] / ksh21["AKTÍVKORÚ21"] * 100
ksh21["RGYK_ARÁNY21"] = ksh21["RGYK21"] / ksh21["NÉPESSÉG21"] * 100
ksh21["AUTÓ_EZER21"] = ksh21["AUTÓ21"] / ksh21["NÉPESSÉG21"] * 1000
ksh21["VÁNDORLÁS_EZER21"] = (ksh21["ODAVÁND21"]-ksh21["ELVÁND21"]) / ksh21["NÉPESSÉG21"] * 1000
ksh21["LN_NÉPSŰRŰSÉG21"] = np.log(nepsuruseg21)
ksh21["LN_NÉPESSÉG21"] = np.log(ksh21["NÉPESSÉG21"])

ksh25["GYEREK_ARÁNY25"] = ksh25["GYEREK25"] / ksh25["NÉPESSÉG25"] * 100
ksh25["IDŐS_ARÁNY25"] = ksh25["IDŐS25"] / ksh25["NÉPESSÉG25"] * 100
ksh25["ÁLLÁSKERESŐ_ARÁNY25"] = ksh25["ÁLLÁSKERESŐ25"] / ksh25["AKTÍVKORÚ25"] * 100
ksh25["KÖZFOGL_ARÁNY25"] = ksh25["KÖZFOGL25"] / ksh25["AKTÍVKORÚ25"] * 100
ksh25["RGYK_ARÁNY25"] = ksh25["RGYK25"] / ksh25["NÉPESSÉG25"] * 100
ksh25["AUTÓ_EZER25"] = ksh25["AUTÓ25"] / ksh25["NÉPESSÉG25"] * 1000
ksh25["VÁNDORLÁS_EZER25"] = (ksh25["ODAVÁND25"]-ksh25["ELVÁND25"]) / ksh25["NÉPESSÉG25"] * 1000

# Ellenőrzés: az új oszlopok (19-től) hiánytalanok
print(ksh21.iloc[:, 19:].isnull().sum().sum(), ksh25.iloc[:, 19:].isnull().sum().sum())   # 0 0

# Kihagyott változók: a döntéseket alátámasztó számok (lásd 3.6. alfejezet)
# Népesség és népsűrűség (log): együttmozgás
print(ksh21["LN_NÉPESSÉG21"].corr(ksh21["LN_NÉPSŰRŰSÉG21"]))   # 0,80

# Vándorlás: mennyire stabil évről évre (a 2021-es és a 2025-ös érték korrelációja)
vandorlas = pd.merge(ksh21[["TELEPÜLÉS", "NÉPESSÉG21", "VÁNDORLÁS_EZER21"]],
                     ksh25[["TELEPÜLÉS", "VÁNDORLÁS_EZER25"]], on="TELEPÜLÉS")
print(vandorlas["VÁNDORLÁS_EZER21"].corr(vandorlas["VÁNDORLÁS_EZER25"]))   # 0,14

kicsi = vandorlas[vandorlas["NÉPESSÉG21"] < 500]
print(kicsi["VÁNDORLÁS_EZER21"].corr(kicsi["VÁNDORLÁS_EZER25"]))   # 0,08

nagy = vandorlas[vandorlas["NÉPESSÉG21"] > 5000]
print(nagy["VÁNDORLÁS_EZER21"].corr(nagy["VÁNDORLÁS_EZER25"]))   # 0,53

# 4.7 Jövedelem: NAV SZJA-alap (TeIR), 2021 és 2024
szja = pd.read_excel("TEIR/TEIR_EGY_LAKOSRA_SZJA_ALAPOT_KEPEZO_BELFOLDI_JOVEDELEM.xlsx", skipfooter=160)
szja.columns = ["TELEPÜLÉS", "KSH_KÓD", "SZJA21", "SZJA24"]

# A területváltozást jelölő „ *” levágása a nevek végéről
print(szja["TELEPÜLÉS"].str.contains("*", regex=False).sum())   # 159
szja["TELEPÜLÉS"] = szja["TELEPÜLÉS"].str.replace(" *", "", regex=False)

# Kerületnevek a 4.4-es szótárral
szja["TELEPÜLÉS"] = szja["TELEPÜLÉS"].replace(kerulet_nevek)
print(szja["TELEPÜLÉS"].isin(ksh21["TELEPÜLÉS"]).sum())   # 3177: név szerint összekapcsolható

# 4.8 A magyarázóváltozók összekapcsolása a választási táblával
# A modellhez szükséges oszlopok (vándorlás, szobaszám és ln népesség nélkül)
# + FHT_ARÁNY21, de csak az 5.3-as VIF-ellenőrzéshez, a modellekbe nem kerül
ossze_ksh21 = ksh21[["TELEPÜLÉS", "KSH_TELEPÜLÉS", "NÉPESSÉG21", "GYEREK_ARÁNY21", "IDŐS_ARÁNY21",
                      "ÁLLÁSKERESŐ_ARÁNY21", "KÖZFOGL_ARÁNY21", "FHT_ARÁNY21", "RGYK_ARÁNY21",
                      "AUTÓ_EZER21", "LN_NÉPSŰRŰSÉG21"]]
ossze_ksh25 = ksh25[["TELEPÜLÉS", "NÉPESSÉG25", "GYEREK_ARÁNY25", "IDŐS_ARÁNY25",
                      "ÁLLÁSKERESŐ_ARÁNY25", "KÖZFOGL_ARÁNY25", "RGYK_ARÁNY25",
                      "AUTÓ_EZER25"]]

telep = pd.merge(telep, ossze_ksh21, how="left", on="TELEPÜLÉS")
telep = pd.merge(telep, ossze_ksh25, how="left", on="TELEPÜLÉS")
telep = pd.merge(telep, szja, how="left", on="TELEPÜLÉS")

print(len(telep))                     # 3177: nincs duplikált név
print(telep.isnull().sum().sum())     # 23: csak a kerületek KSH-kódja üres

# A név szerint behozott jövedelemadat a KSH-kód szerint is a jó településé
print((telep["KSH_TELEPÜLÉS"].str[:4].astype(float) == telep["KSH_KÓD"]).sum())   # 3154

# 4.9 Változások 2021 → 2025 (jövedelem: 2021 → 2024)
# Különbségek (arányoknál százalékpont, autónál darab/ezer fő)
telep["D_GYEREK_ARÁNY"] = telep["GYEREK_ARÁNY25"] - telep["GYEREK_ARÁNY21"]
telep["D_IDŐS_ARÁNY"] = telep["IDŐS_ARÁNY25"] - telep["IDŐS_ARÁNY21"]
telep["D_ÁLLÁSKERESŐ_ARÁNY"] = telep["ÁLLÁSKERESŐ_ARÁNY25"] - telep["ÁLLÁSKERESŐ_ARÁNY21"]
telep["D_KÖZFOGL_ARÁNY"] = telep["KÖZFOGL_ARÁNY25"] - telep["KÖZFOGL_ARÁNY21"]
telep["D_RGYK_ARÁNY"] = telep["RGYK_ARÁNY25"] - telep["RGYK_ARÁNY21"]
telep["D_AUTÓ_EZER"] = telep["AUTÓ_EZER25"] - telep["AUTÓ_EZER21"]

# Népességváltozás (%)
telep["NÉPESSÉG_VÁLT"] = (telep["NÉPESSÉG25"] / telep["NÉPESSÉG21"] - 1) * 100

# Reáljövedelem-változás (%)
# Deflátor 2021 → 2024: a KSH fogyasztóiár-indexe (előző év = 100), 2022: 114,5; 2023: 117,6; 2024: 103,7
# Forrás: KSH STADAT 1.2.1.2. tábla (ara0040), éves adat (január–december, előző év = 100)
deflator = 1.145 * 1.176 * 1.037
telep["REÁLJÖV_NÖV"] = (telep["SZJA24"] / deflator / telep["SZJA21"] - 1) * 100

# A jövedelem szintje ezer forintban (a β így 1000 Ft/fő különbségre értelmezhető)
telep["SZJA21_EFT"] = telep["SZJA21"] / 1000

uj_oszlopok = ["D_GYEREK_ARÁNY", "D_IDŐS_ARÁNY", "D_ÁLLÁSKERESŐ_ARÁNY", "D_KÖZFOGL_ARÁNY",
               "D_RGYK_ARÁNY", "D_AUTÓ_EZER", "NÉPESSÉG_VÁLT", "REÁLJÖV_NÖV", "SZJA21_EFT"]

print(telep[uj_oszlopok].isnull().sum().sum())   # 0
print(telep["REÁLJÖV_NÖV"].median())             # 12,33

# Deflátor nélküli, nominális medián
print(((telep["SZJA24"] / telep["SZJA21"] - 1) * 100).median())   # 56,9


# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 5. REGRESSZIÓS MODELLEK - SÚLYOZOTT OLS (WLS)
# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# 5.1 Első modell: csak a változások (ΔX)
dx = ["D_GYEREK_ARÁNY", "D_IDŐS_ARÁNY", "D_ÁLLÁSKERESŐ_ARÁNY", "D_KÖZFOGL_ARÁNY",
      "D_RGYK_ARÁNY", "D_AUTÓ_EZER", "NÉPESSÉG_VÁLT", "REÁLJÖV_NÖV"]
y = telep["FIDESZ_VÁLTOZÁS"]
X1 = sm.add_constant(telep[dx])
modell1 = sm.WLS(y, X1, weights=telep["ÉRVÉNYES22"]).fit(cov_type="HC1")
print(modell1.summary())

# 5.2 Második modell: változások + 2021-es kiinduló szintek
# A szintek kontrollként: a változás és a Fidesz-változás is a település
# kiinduló helyzetéből fakadhat (confounding)
szintek = ["GYEREK_ARÁNY21", "IDŐS_ARÁNY21", "ÁLLÁSKERESŐ_ARÁNY21",
           "KÖZFOGL_ARÁNY21", "RGYK_ARÁNY21", "AUTÓ_EZER21",
           "LN_NÉPSŰRŰSÉG21", "SZJA21_EFT"]

X2 = sm.add_constant(telep[dx + szintek])
modell2 = sm.WLS(y, X2, weights=telep["ÉRVÉNYES22"]).fit(cov_type="HC1")
print(modell2.summary())

# 5.3 Multikollinearitás: a magyarázóváltozók egymás közötti összefüggése
# VIF = 1 / (1 - R²), R²: a változó regressziója az összes többi magyarázóváltozóra
def vif_tabla(X):
    vif = []
    for i in range(len(X.columns)):
        vif.append(variance_inflation_factor(X, i))
    return pd.DataFrame({"VÁLTOZÓ": X.columns, "VIF": vif})

# A 2. modell VIF-értékei (a const sora nem értelmezendő)
print(vif_tabla(X2))

# Ellenőrzés kézzel egy változóra (súlyozatlan OLS, mert a VIF-függvény is súlyozatlan)
r2 = sm.OLS(telep["ÁLLÁSKERESŐ_ARÁNY21"], X2.loc[:, ~X2.columns.isin(["ÁLLÁSKERESŐ_ARÁNY21"])]).fit().rsquared
print(1 / (1 - r2))   # 4,03 (egyezik a fenti VIF-táblával)

# Összevetés: ugyanez az FHT-val együtt (ez indokolja az FHT kihagyását, lásd 3.6)
X2_fht = sm.add_constant(telep[dx + szintek + ["FHT_ARÁNY21"]])
print(vif_tabla(X2_fht))   # FHT_ARÁNY21 4,06, ÁLLÁSKERESŐ_ARÁNY21 7,06, RGYK_ARÁNY21 7,64

# 5.4 Harmadik modell: változások + 2021-es kiinduló szintek + 2022-es Fidesz-arány
X3 = sm.add_constant(telep[dx + szintek + ["FIDESZ_ARÁNY22"]])

modell3 = sm.WLS(y, X3, weights=telep["ÉRVÉNYES22"]).fit(cov_type="HC1")
print(modell3.summary())

# 5.5 Multikollinearitás a 3. modellben
print(vif_tabla(X3))

# 5.6 Összesítő táblázat a három modellről
# A táblázat aljára kerülő megfigyelésszám (N) egy modellre
def megfigyelesek(modell):
    return str(int(modell.nobs))

tabla = summary_col([modell1, modell2, modell3],
                    model_names=["(1)", "(2)", "(3)"],
                    stars=True, float_format="%.4f",
                    regressor_order=dx + szintek + ["FIDESZ_ARÁNY22"],
                    info_dict={"N": megfigyelesek})
print(tabla)
tabla.tables[0].to_excel("Export/modellek.xlsx")

# 5.7 Leíró statisztika a modell változóira (súlyozatlan)
leiro = telep[["FIDESZ_VÁLTOZÁS"] + dx + szintek + ["FIDESZ_ARÁNY22"]].describe().T
leiro.to_excel("Export/leiro.xlsx")

# A legnagyobb Fidesz-növekedés és -csökkenés települése
oszlopok = ["TELEPÜLÉS", "NÉPESSÉG21", "ÉRVÉNYES22", "ÉRVÉNYES26", "FIDESZ_ARÁNY22", "FIDESZ_ARÁNY26", "FIDESZ_VÁLTOZÁS"]
print(telep.loc[telep["FIDESZ_VÁLTOZÁS"].idxmax(), oszlopok])   # Szárász
print(telep.loc[telep["FIDESZ_VÁLTOZÁS"].idxmin(), oszlopok])   # Nyésta

# Az eredményváltozó ferdesége
print(telep["FIDESZ_VÁLTOZÁS"].skew())   # −0,79

# A Fidesz-arányban és a részvételben kivételes települések mérete (lakónépesség, 2021)
print(telep[telep["FIDESZ_VÁLTOZÁS"] >= 0]["NÉPESSÉG21"].max())   # 350
print(telep[telep["RÉSZVÉTEL26"] - telep["RÉSZVÉTEL22"] <= 0]["NÉPESSÉG21"].median())   # 192
print(telep["NÉPESSÉG21"].median())   # 787

# 5.8 A 2022-es Fidesz-arány és a Fidesz-változás, településméret szerint (2. ábra)
# Méretkategóriák a 2021-es lakónépesség alapján
# (határok: 500 fő – a 6. és 7. modell határa, 5000 fő)
print((telep["NÉPESSÉG21"] == 5000).sum())   # 0: nincs pontosan 5000 fős
kategoriak = ["500 fő alatt", "500–4999 fő", "5000 fő felett", "Budapesti kerület"]
telep["MÉRET_KAT"] = pd.cut(telep["NÉPESSÉG21"], bins=[0, 500, 5000, np.inf],
                            labels=kategoriak[:3], right=False).astype(str)
telep.loc[telep["TELEPÜLÉS"].str.startswith("Budapest"), "MÉRET_KAT"] = "Budapesti kerület"
print(telep["MÉRET_KAT"].value_counts()[kategoriak])   # ellenőrzés: az 500 alattiak száma 3177 − 2021 = 1156

szinek = ["#C6DBF5", "#5E97DE", "#15468A", "#081428"]

fig, ax = plt.subplots(figsize=(6.3, 4.8))
# Sorban rajzolva: a kisebb kategóriák alulra, a nagyobbak felülre kerülnek
for kategoria, szin_kat in zip(kategoriak, szinek):
    resz = telep[telep["MÉRET_KAT"] == kategoria]
    ax.scatter(resz["FIDESZ_ARÁNY22"], resz["FIDESZ_VÁLTOZÁS"], s=10,
               color=szin_kat, alpha=0.7, edgecolor="none",
               label=kategoria + " (" + str(len(resz)) + ")")

ax.axhline(0, color="#999999", linewidth=0.8)
ax.set_xticks(range(20, 101, 10))
ax.set_yticks(range(-70, 21, 10))
ax.set_xlim(20, 100.5)
ax.set_ylim(-70, 20)
ax.set_xlabel("Fidesz-arány 2022-ben (%)")
ax.set_ylabel("A Fidesz-arány változása 2022 és 2026 között\n(százalékpont)")
ax.legend(title="Lakónépesség (2021. év vége)", loc="lower left", markerscale=2, frameon=False)
ax.grid(color="#DDDDDD", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("ÁBRÁK/fidesz22_valtozas.png", dpi=300, bbox_inches="tight")
plt.show()


# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡
# 6. ROBUSZTUSSÁG
# ≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡≡

# (a 3. modell változatai)
# 6.1 Súlyozatlanul
modell4 = sm.OLS(y, X3).fit(cov_type="HC1")

# 6.2 A 2026-os érvényes szavazatokkal súlyozva
modell5 = sm.WLS(y, X3, weights=telep["ÉRVÉNYES26"]).fit(cov_type="HC1")

# 6.3 Az 500 fő alatti települések nélkül (a legzajosabb arányok)
telep500 = telep[telep["NÉPESSÉG21"] >= 500]
print(len(telep500))   # 2021

y6 = telep500["FIDESZ_VÁLTOZÁS"]
X6 = sm.add_constant(telep500[dx + szintek + ["FIDESZ_ARÁNY22"]])
modell6 = sm.WLS(y6, X6, weights=telep500["ÉRVÉNYES22"]).fit(cov_type="HC1")

# 6.4 Súlyozatlanul, 500 fő felett
modell7 = sm.OLS(y6, X6).fit(cov_type="HC1")

# 6.5 Megye-dummykkal (regionális különbségek kiszűrése)
# Referencia: Budapest → 19 dummy
megye_d = pd.get_dummies(telep["MEGYE"], dtype=int).drop(columns="BUDAPEST")
X8 = pd.concat([X3, megye_d], axis=1)
modell8 = sm.WLS(y, X8, weights=telep["ÉRVÉNYES22"]).fit(cov_type="HC1")

# 6.6 Összesítő táblázat: a 3. modell és változatai
tabla2 = summary_col([modell3, modell4, modell5, modell6, modell7, modell8],
                     model_names=["(3)", "(4)", "(5)", "(6)", "(7)", "(8)"],
                     stars=True, float_format="%.4f",
                     regressor_order=dx + szintek + ["FIDESZ_ARÁNY22", "const"],
                     drop_omitted=True,   # a modell8 megye-dummyjai nem kerülnek a táblába
                     info_dict={"N": megfigyelesek})
print(tabla2)
tabla2.tables[0].to_excel("Export/robusztussag.xlsx")