import requests
import time
import pandas as pd
from   datetime import datetime
import numpy as np
import pytz
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from   matplotlib.ticker import FuncFormatter
import statsmodels.api as sm

file_path_intra = 'SPY_intra_data_2022-05-09_to_2024-04-22.csv'
file_path_daily = 'SPY_daily_data_2022-05-09_to_2024-04-22.csv'
file_path_dividends = 'SPY_dividends_2022-05-09_to_2024-04-22.csv'
# Load the intraday data into a DataFrame and set the datetime column as the index.
df = pd.read_csv(file_path_intra)
#df.info()
df['caldt'] = pd.to_datetime(df['caldt'], errors='coerce')
df['day'] = pd.to_datetime(df['caldt'], errors='coerce').dt.date  # Extract the date part from the datetime for daily analysis.
df.set_index('caldt', inplace=True)  # Setting the datetime as the index for easier time series manipulation.

#df.info()
# Group the DataFrame by the 'day' column to facilitate operations that need daily aggregation.
daily_groups = df.groupby('day')

# Extract unique days from the dataset to iterate through each day for processing.
all_days = df['day'].unique()

# Initialize new columns to store calculated metrics, starting with NaN for absence of initial values.
df['move_open'] = np.nan  # To record the absolute daily change from the open price
df['vwap'] = np.nan       # To calculate the Volume Weighted Average Price.
df['spy_dvol'] = np.nan   # To record SPY's daily volatility.
#df.info()
# Create a series to hold computed daily returns for SPY, initialized with NaN.
spy_ret = pd.Series(index=all_days, dtype=float)

# Iterate through each day to calculate metrics.
for d in range(1, len(all_days)):
    current_day = all_days[d]
    prev_day = all_days[d - 1]

    # Access the data for the current and previous days using their groups.
    current_day_data = daily_groups.get_group(current_day)
    prev_day_data = daily_groups.get_group(prev_day)

    # Calculate the average of high, low, and close prices.
    hlc = (current_day_data['high'] + current_day_data['low'] + current_day_data['close']) / 3

    # Compute volume-weighted metrics for VWAP calculation.
    vol_x_hlc = current_day_data['volume'] * hlc
    cum_vol_x_hlc = vol_x_hlc.cumsum()  # Cumulative sum for VWAP calculation.
    cum_volume = current_day_data['volume'].cumsum()

    # Assign the calculated VWAP to the corresponding index in the DataFrame.
    df.loc[current_day_data.index, 'vwap'] = cum_vol_x_hlc / cum_volume

    # Calculate the absolute percentage change from the day's opening price.
    open_price = current_day_data['open'].iloc[0]
    df.loc[current_day_data.index, 'move_open'] = (current_day_data['close'] / open_price - 1).abs()

    # Compute the daily return for SPY using the closing prices from the current and previous day.
    spy_ret.loc[current_day] = current_day_data['close'].iloc[-1] / prev_day_data['close'].iloc[-1] - 1

    # Calculate the 15-day rolling volatility, starting calculation after accumulating 15 days of data.
    if d > 14:
        df.loc[current_day_data.index, 'spy_dvol'] = spy_ret.iloc[d - 15:d - 1].std(skipna=False)

# Calculate the minutes from market open and determine the minute of the day for each timestamp.
df['min_from_open'] = ((df.index - df.index.normalize()) / pd.Timedelta(minutes=1)) - (9 * 60 + 30) + 1
df['minute_of_day'] = df['min_from_open'].round().astype(int)

# Group data by 'minute_of_day' for minute-level calculations.
minute_groups = df.groupby('minute_of_day')

# Calculate rolling mean and delayed sigma for each minute of the trading day.
df['move_open_rolling_mean'] = minute_groups['move_open'].transform(lambda x: x.rolling(window=14, min_periods=13).mean())
df['sigma_open'] = minute_groups['move_open_rolling_mean'].transform(lambda x: x.shift(1))

'''
print(f"\n--- Vérification de l'index de sigma_open pour {current_day} ---")
print(f"Type d'index de sigma_open : { current_day_data.index.dtype}")
'''
# Convert dividend dates to datetime and merge dividend data based on trading days.
dividends = pd.read_csv(file_path_dividends)
dividends['day'] = pd.to_datetime(dividends['caldt']).dt.date
df = df.reset_index().merge(dividends[['day', 'dividend']], on='day', how='left').set_index('caldt')#df = df.merge(dividends[['day', 'dividend']], on='day', how='left')
df['dividend'] = df['dividend'].fillna(0)  # Fill missing dividend data with 0.




#df.info()




# Group data by day for faster access
AUM_0 = 100000.0   # Initial amount of money to manage
commission = 0.0035  # Commission rate per trade
min_comm_per_order = 0.35  # Minimum commission per trade
band_mult = 1  # Multiplier for the trading band
trade_freq = 30  # Frequency of trades in minutes
sizing_type = "vol_target"  # Strategy for sizing positions based on volatility "vol_target"/ "full_notional"
target_vol = 0.02  # Target volatility for the position sizing
max_leverage = 4  # Maximum leverage allowed

# Group data by day for faster access
daily_groups = df.groupby('day')

# Initialize strategy DataFrame using unique days
strat = pd.DataFrame(index=all_days)
strat['ret'] = np.nan
strat['AUM'] = AUM_0
strat['ret_spy'] = np.nan

df_daily = pd.read_csv(file_path_daily)
df_daily['caldt'] = pd.to_datetime(df_daily['caldt'],errors='coerce')
#df_daily.info()
df_daily.set_index('caldt', inplace=True)
df_daily['ret'] = df_daily['close'].diff() / df_daily['close'].shift()



liste_UB,liste_LB=[],[]
dict_exposure = {}

# Iterate through each day to calculate metrics.
for d in range(1,len(all_days)):
    current_day = all_days[d]
    prev_day = all_days[d - 1]
    #print(f'\n--- Début des calculs pour le jour: {current_day} (d={d}) ---') # Ajoute l'info sur le jour

    if prev_day in daily_groups.groups and current_day in daily_groups.groups:
        prev_day_data = daily_groups.get_group(prev_day)
        current_day_data = daily_groups.get_group(current_day)

        if 'sigma_open' in current_day_data.columns and current_day_data['sigma_open'].isna().all():
            #print(f"DEBUG: Skipping {current_day} because sigma_open is all NaN.")
            continue # Passe au jour suivant si sigma_open est tout NaN

        prev_close_adjusted = prev_day_data['close'].iloc[-1] - df.loc[current_day_data.index, 'dividend'].iloc[-1]
        open_price = current_day_data['open'].iloc[0]
        current_close_prices = current_day_data['close']
        spx_vol = current_day_data['spy_dvol'].iloc[0]
        vwap = current_day_data['vwap']


        sigma_open_from_slice = current_day_data['sigma_open']
        sigma_open = df.loc[current_day_data.index, 'sigma_open']


        UB = max(open_price, prev_close_adjusted) * (1 + band_mult * sigma_open)
        LB = min(open_price, prev_close_adjusted) * (1 - band_mult * sigma_open)

        liste_UB.append(UB)
        liste_LB.append(LB)

        # Determine trading signals
        signals = np.zeros_like(current_close_prices)
        signals[(current_close_prices > UB) & (current_close_prices > vwap)] = 1
        signals[(current_close_prices < LB) & (current_close_prices < vwap)] = -1
        #signals[(current_close_prices > UB)] = 1
        #signals[(current_close_prices < LB)] -= 1
        #signals[current_close_prices > vwap] = 1
        #signals[current_close_prices < vwap] = -1



        # Position sizing
        previous_aum = strat.loc[prev_day, 'AUM']

        if sizing_type == "vol_target":
            if math.isnan(spx_vol):
                shares = round(previous_aum / open_price * max_leverage)
            else:
                shares = round(previous_aum / open_price * min(target_vol / spx_vol, max_leverage))

        elif sizing_type == "full_notional":
            shares = round(previous_aum / open_price)

        # Apply trading signals at trade frequencies
        trade_indices = np.where(current_day_data["min_from_open"] % trade_freq == 0)[0]
        exposure = np.full(len(current_day_data), np.nan)  # Start with NaNs
        exposure[trade_indices] = signals[trade_indices]  # Apply signals at trade times

        # Custom forward-fill that stops at zeros
        last_valid = np.nan  # Initialize last valid value as NaN
        filled_values = []  # List to hold the forward-filled values
        for value in exposure:
            if not np.isnan(value):  # If current value is not NaN, update last valid value
                last_valid = value
            if last_valid == 0:  # Reset if last valid value is zero
                last_valid = np.nan
            filled_values.append(last_valid)

        exposure = pd.Series(filled_values, index=current_day_data.index).shift(1).fillna(
            0).values  # Apply shift and fill NaNs
        # --- LA FERMETURE EN FIN DE JOURNÉE ---

        #if exposure.size > 0:  # S'assurer qu'il y a des données d'exposition pour la journée
          #  last_minute_index_in_exposure = -1  # Le dernier élément de l'array NumPy
            # Si la position à la dernière minute n'est pas déjà zéro
         #   if exposure[last_minute_index_in_exposure] != 0:
           #     exposure[last_minute_index_in_exposure] = 0  # Force la position à zéro

        #print(current_day)
        #print(dict_exposure)
        dict_exposure[current_day] = exposure

        # Calculate trades count based on changes in exposure
        trades_count = np.sum(np.abs(np.diff(np.append(exposure, 0))))

        # Calculate PnL
        change_1m = current_close_prices.diff()
        gross_pnl = np.sum(exposure * change_1m) * shares
        commission_paid = trades_count * max(min_comm_per_order, commission * shares)
        net_pnl = gross_pnl - commission_paid


        # Update the daily return and new AUM
        strat.loc[current_day, 'AUM'] = previous_aum + net_pnl
        strat.loc[current_day, 'ret'] = net_pnl / previous_aum
        # Save the passive Buy&Hold daily return for SPY
        strat.loc[current_day, 'ret_spy'] = df_daily.loc[df_daily.index.date == current_day, 'ret'].values[0]
#print(strat[ 'AUM'] )
#print(strat[ 'ret'])
#print(strat['ret_spy'])
#print(dict_exposure)


from matplotlib.ticker import FuncFormatter # Nécessaire pour le formatage de l'axe Y de l'AUM global

# --- DÉFINIR LA JOURNÉE À TRACER ---
dates_to_plot_str = ['2023-09-05','2023-09-06','2023-09-07','2023-09-08','2023-09-09','2023-09-10','2023-09-11','2023-09-12'


] # Modifie cette date
''' 
,'2023-12-18',
'2023-12-19',
'2023-12-20'
,'2023-12-22'
,'2023-12-29'
'''

for day in dates_to_plot_str:
    day_to_plot = pd.to_datetime(day).date()
    try:
        # Récupère les données de la journée
        current_day_data_for_plot = daily_groups.get_group(day_to_plot)

        # --- Récupération des données pour les lignes de référence ---
        open_price = current_day_data_for_plot['open'].iloc[0]

        idx_in_all_days = None
        for i, day in enumerate(all_days):
            if day == day_to_plot:
                idx_in_all_days = i
                break

        if idx_in_all_days is None or idx_in_all_days == 0:
            raise ValueError(f"Le jour {day_to_plot} n'a pas été trouvé ou est le premier jour (pas de données du jour précédent).")

        prev_day = all_days[idx_in_all_days - 1]
        prev_close_adjusted = float('nan') # Initialisation
        if prev_day in daily_groups.groups:
            prev_day_data = daily_groups.get_group(prev_day)
            # Assure-toi que 'df' est accessible et contient 'dividend'
            dividend_for_current_day = df.loc[current_day_data_for_plot.index[0], 'dividend']
            prev_close_adjusted = prev_day_data['close'].iloc[-1] - dividend_for_current_day

        # --- Récupération des Bandes (via listes) et de l'Exposition (via dictionnaire) ---
        ub_series_for_plot = liste_UB[idx_in_all_days - 14] # Conserve ton indexation existante
        lb_series_for_plot = liste_LB[idx_in_all_days - 14] # Conserve ton indexation existante

        # Convertir l'exposition en pd.Series au cas où elle aurait été stockée en NumPy array
        day_exposure_series = pd.Series(dict_exposure[day_to_plot], index=current_day_data_for_plot.index)

        # --- Configuration du Graphique (UN SEUL AXE Y) ---
        fig, ax1 = plt.subplots(figsize=(14, 7)) # Il n'y a plus qu'un seul axe

        # --- Axe Y Principal (Prix) ---
        ax1.plot(current_day_data_for_plot.index, current_day_data_for_plot['close'],
                 label='Prix Clôture Intraday SPY', color='blue', linewidth=1.5)
        ax1.plot(ub_series_for_plot.index, ub_series_for_plot,
                 label='Bande Supérieure (UB)', color='red', linestyle='--', linewidth=1)
        ax1.plot(lb_series_for_plot.index, lb_series_for_plot,
                 label='Bande Inférieure (LB)', color='green', linestyle='--', linewidth=1) # Couleur changée en vert, comme d'habitude

        if 'vwap' in current_day_data_for_plot.columns and current_day_data_for_plot['vwap'].notna().any():
            ax1.plot(current_day_data_for_plot.index, current_day_data_for_plot['vwap'],
                     label='VWAP', color='black', linestyle='-.', linewidth=0.8)

        # Lignes Open et Previous Close Adjusted
        ax1.axhline(y=open_price, color='purple', linestyle=':', linewidth=0.8, label=f'Prix Ouverture SPY ({open_price:.2f})')


        # Ajout des marqueurs d'ordre
        exposure_changes = day_exposure_series.diff().fillna(0)
        buy_signals_idx = exposure_changes[exposure_changes > 0].index
        sell_signals_idx = exposure_changes[exposure_changes < 0].index
        prev_exposure = day_exposure_series.shift(1).fillna(0)
        #exit_signals_idx = day_exposure_series[(day_exposure_series == 0) & ((prev_exposure == 1) | (prev_exposure == -1))].index
        print( buy_signals_idx)
        print(sell_signals_idx)
        #print(exit_signals_idx )



        if not buy_signals_idx.empty:
            buy_prices = current_day_data_for_plot.loc[buy_signals_idx, 'close']
            ax1.plot(buy_signals_idx, buy_prices, '^', markersize=10, color='green', label='Ordre Achat')
        if not sell_signals_idx.empty:
            sell_prices = current_day_data_for_plot.loc[sell_signals_idx, 'close']
            ax1.plot(sell_signals_idx, sell_prices, 'v', markersize=10, color='red', label='Ordre Vente')
        ''' 
        if not exit_signals_idx.empty:
            exit_prices = current_day_data_for_plot.loc[exit_signals_idx, 'close']
            ax1.plot(exit_signals_idx, exit_prices, 'o', markersize=8, color='gold', fillstyle='none', label='Sortie Position')
        '''
        ax1.set_xlabel('Heure de la Journée')
        ax1.set_ylabel('Prix') # Pas de couleur spécifique si un seul axe Y
        ax1.tick_params(axis='y') # Pas de couleur spécifique si un seul axe Y
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper left')

        # --- Configuration Générale du Graphique ---
        # Récupérer l'AUM et le PnL journalier pour le jour à tracer
        # Assure-toi que 'strat' est entièrement rempli avant cette ligne
        aum_for_title = strat.loc[day_to_plot, 'AUM']
        daily_pnl_percent = strat.loc[day_to_plot, 'ret'] * 100 # Convertir en pourcentage

        plt.title(f'Prix, Bandes & Ordres pour SPY le {day_to_plot.strftime("%Y-%m-%d")} | AUM: ${aum_for_title:,.2f} | PnL Jour: {daily_pnl_percent:+.2f}%') # <-- MODIFICATION DU TITRE

        locator = mdates.MinuteLocator(interval=30)
        formatter = mdates.DateFormatter('%H:%M')
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Une erreur est survenue lors du tracé pour le jour {day_to_plot} : {e}")
        print("Vérifiez que le jour choisi est présent dans vos données et que les bandes et l'exposition ont été calculées pour ce jour.")

# --- Garde cette partie en dehors du bloc try/except de tracé intraday ---
# Elle concerne le graphique global de l'AUM de la stratégie
# Calculate cumulative products for AUM calculations
strat['AUM_SPX'] = AUM_0 * (1 + strat['ret_spy']).cumprod(skipna=True)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plotting the AUM of the strategy and the passive S&P 500 exposure
ax.plot(strat.index, strat['AUM'], label='Momentum', linewidth=2, color='k')
ax.plot(strat.index, strat['AUM_SPX'], label='S&P 500', linewidth=1, color='r')

# Formatting the plot
ax.grid(True, linestyle=':')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_ylabel('AUM ($)')
plt.legend(loc='upper left')
plt.title('Intraday Momentum Strategy', fontsize=12, fontweight='bold')
plt.suptitle(f'Commission = ${commission}/share', fontsize=9, verticalalignment='top')

# Show the plot
plt.show()

# Calculate additional stats and display them

first_close_price = df['close'].iloc[0]
last_close_price = df['close'].iloc[-1]
buy_and_hold_return = (last_close_price - first_close_price) / first_close_price
Y = strat['ret'].dropna()
X = sm.add_constant(strat['ret_spy'].dropna())
model = sm.OLS(Y, X).fit()
stats = {

    'Buy and Hold Return (%)':  round(buy_and_hold_return * 100, 2),
    'Total Return (%)': round((np.prod(1 + strat['ret'].dropna()) - 1) * 100, 2),
    'Annualized Return (%)': round((np.prod(1 + strat['ret']) ** (252 / len(strat['ret'])) - 1) * 100, 1),
    'Annualized Volatility (%)': round(strat['ret'].dropna().std() * np.sqrt(252) * 100, 1),
    'Sharpe Ratio': round(strat['ret'].dropna().mean() / strat['ret'].dropna().std() * np.sqrt(252), 2),
    'Hit Ratio (%)': round((strat['ret'] > 0).sum() / (strat['ret'].abs() > 0).sum() * 100, 0),
    'Maximum Drawdown (%)': round(strat['AUM'].div(strat['AUM'].cummax()).sub(1).min() * -100, 0),
    'Alpha (%)':round(model.params.const * 100 * 252, 2),
    'Beta': round(model.params['ret_spy'], 3),
}


fichier_performance = "strategy_metrics.txt"

with open(fichier_performance, 'a') as f:
    string_stats = str(stats)
    f.write("---New SPY Momentum Strategies---\n")
    f.write(string_stats)
    f.write("\n")




print(stats)



'''            
# --- DÉBUT DU TRACÉ DES GRAPHIQUES ---

# Créer une figure avec deux sous-graphes, l'un au-dessus de l'autre
# sharex=True pour que les axes des abscisses soient alignés et zooment ensemble
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- Graphique du Haut : AUM (Asset Under Management) ---
ax1.plot(strat.index, strat['AUM'], label='AUM Stratégie', color='darkblue', linewidth=2)
ax1.plot(strat.index, strat['AUM_SPX'], label='AUM S&P 500 (Benchmark)', color='darkred', linewidth=1.5, linestyle='--') # Optionnel: pour comparer AUM vs Benchmark

ax1.set_title('Performance de la Stratégie (AUM vs Benchmark)')
ax1.set_ylabel('Valeur du Capital ($)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper left')

# Formatte l'axe Y de l'AUM en format monétaire
formatter_money = FuncFormatter(lambda x, p: f'${x:,.0f}')
ax1.yaxis.set_major_formatter(formatter_money)


# --- Graphique du Bas : Rendements Quotidiens ---
ax2.plot(strat.index, strat['ret'] * 100, label='Rendement Quotidien Stratégie (%)', color='green', linewidth=0.8)
ax2.plot(strat.index, strat['ret_spy'] * 100, label='Rendement Quotidien SPY (%)', color='orange', linewidth=0.8, linestyle=':')

ax2.set_xlabel('Date')
ax2.set_ylabel('Rendement (%)')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend(loc='upper left')

# Formatte l'axe Y des rendements en pourcentage
formatter_percent = FuncFormatter(lambda y, p: f'{y:.2f}%')
ax2.yaxis.set_major_formatter(formatter_percent)

# --- Configuration de l'axe X (partagé) ---
# Utilise un localisateur et un formateur pour les dates, par exemple par mois ou année
locator = mdates.MonthLocator(interval=3) # Tous les 3 mois
formatter = mdates.DateFormatter('%b %Y') # Mois Année (ex: Jan 2020)

ax2.xaxis.set_major_locator(locator)
ax2.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=45) # Rotation pour éviter que les dates ne se chevauchent

# Ajuste l'espacement entre les sous-graphes
plt.tight_layout()

# Affiche le graphique
plt.show()
'''