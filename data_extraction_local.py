import pandas as pd
import os
from apiData import fetch_polygon_data,fetch_polygon_dividends




ticker = 'SPY'
from_date = '2022-05-09'
until_date = '2024-04-22'

# --- Définir les noms de fichiers locaux ---
intra_data_filename_csv = f"{ticker}_intra_data_{from_date}_to_{until_date}.csv"
daily_data_filename_csv = f"{ticker}_daily_data_{from_date}_to_{until_date}.csv"
dividends_filename_csv = f"{ticker}_dividends_{from_date}_to_{until_date}.csv"



if os.path.exists(intra_data_filename_csv):
    print(f"Chargement des données intra-journalières depuis {intra_data_filename_csv}...")
    spy_intra_data = pd.read_csv(intra_data_filename_csv)

else:
    print("Données intra-journalières non trouvées localement. Récupération via l'API...")
    spy_intra_data = fetch_polygon_data(ticker, from_date, until_date, 'minute')


    if not spy_intra_data.empty:
        print(f"Sauvegarde des données intra-journalières au format csv dans {intra_data_filename_csv}...")
        spy_intra_data.to_csv(intra_data_filename_csv, index=False)
    else:
        print("Aucune donnée intra-journalière récupérée de l'API.")


if os.path.exists(daily_data_filename_csv):
    print(f"Chargement des données intra-journalières depuis {intra_data_filename_csv}...")
    spy_daily_data = pd.read_csv(daily_data_filename_csv)

else:
    print("Daily data non trouvées localement. Récupération via l'API...")
    spy_daily_data = fetch_polygon_data(ticker, from_date, until_date, 'day')


    if not spy_intra_data.empty:
        print(f"Sauvegarde des dividends au format csv dans {daily_data_filename_csv}...")
        spy_daily_data.to_csv(daily_data_filename_csv, index=False)
    else:
        print("Aucune donnée intra-journalière récupérée de l'API.")


if  os.path.exists(dividends_filename_csv):
    print(f"Chargement des  dividends depuis {dividends_filename_csv}...")
    spy_dividends_data = pd.read_csv(dividends_filename_csv)

else:
    print("Dividendes non trouvées localement. Récupération via l'API...")
    spy_dividends_data = fetch_polygon_dividends(ticker)


    if not spy_dividends_data.empty:
        print(f"Sauvegarde des dividends au format csv dans {dividends_filename_csv}...")
        spy_dividends_data.to_csv(dividends_filename_csv, index=False)
    else:
        print("Aucune donnée intra-journalière récupérée de l'API.")


