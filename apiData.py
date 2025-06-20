import requests
import time
import pandas as pd
from   datetime import datetime
import pytz



API_KEY = ''
BASE_URL = 'https://api.polygon.io'

# Define the rate limit enforcement based on the API tier, Free or Paid.
ENFORCE_RATE_LIMIT = True


def fetch_polygon_data(ticker, start_date, end_date, period, enforce_rate_limit=ENFORCE_RATE_LIMIT):
    """Fetch stock data from Polygon.io based on the given period (minute or day).
       enforce_rate_limit: Set to True to enforce rate limits (suitable for free tiers), False for paid tiers with minimal or no rate limits.
    """
    multiplier = '1'
    timespan = period
    limit = '50000'  # Maximum entries per request
    eastern = pytz.timezone('America/New_York')  # Eastern Time Zone

    url = f'{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}?adjusted=false&sort=asc&limit={limit}&apiKey={API_KEY}'
    #response = requests.get(url)
    #print(response.status_code, response.text)

    data_list = []
    request_count = 0
    first_request_time = None

    while True:
        if enforce_rate_limit and request_count == 5:
            elapsed_time = time.time() - first_request_time
            if elapsed_time < 60:
                wait_time = 60 - elapsed_time
                print(f"API rate limit reached. Waiting {wait_time:.2f} seconds before next request.")
                time.sleep(wait_time)
            request_count = 0
            first_request_time = time.time()  # Reset the timer after the wait

        if first_request_time is None and enforce_rate_limit:
            first_request_time = time.time()

        response = requests.get(url)
        if response.status_code != 200:
            error_message = response.json().get('error', 'No specific error message provided')
            print(f"Error fetching data: {error_message}")
            break

        data = response.json()
        request_count += 1

        results_count = len(data.get('results', []))
        print(f"Fetched {results_count} entries from API.")

        if 'results' in data:
            for entry in data['results']:
                utc_time = datetime.fromtimestamp(entry['t'] / 1000, pytz.utc)
                eastern_time = utc_time.astimezone(eastern)

                data_entry = {
                    'volume': entry['v'],
                    'open': entry['o'],
                    'high': entry['h'],
                    'low': entry['l'],
                    'close': entry['c'],
                    'caldt': eastern_time.replace(tzinfo=None)
                }

                if period == 'minute':
                    if eastern_time.time() >= datetime.strptime('09:30',
                                                                '%H:%M').time() and eastern_time.time() <= datetime.strptime(
                            '15:59', '%H:%M').time():
                        data_list.append(data_entry)
                else:
                    data_list.append(data_entry)

        if 'next_url' in data and data['next_url']:
            url = data['next_url'] + '&apiKey=' + API_KEY
        else:
            break

    df = pd.DataFrame(data_list)
    print("Data fetching complete.")
    return df


def fetch_polygon_dividends(ticker):
    """ Fetches dividend data from Polygon.io for a specified stock ticker. """
    url = f'{BASE_URL}/v3/reference/dividends?ticker={ticker}&limit=1000&apiKey={API_KEY}'

    dividends_list = []
    while True:
        response = requests.get(url)
        data = response.json()
        if 'results' in data:
            for entry in data['results']:
                dividends_list.append({
                    'caldt': datetime.strptime(entry['ex_dividend_date'], '%Y-%m-%d'),
                    'dividend': entry['cash_amount']
                })

        if 'next_url' in data and data['next_url']:
            url = data['next_url'] + '&apiKey=' + API_KEY
        else:
            break

    return pd.DataFrame(dividends_list)

'''
ticker = 'SPY'
from_date = '2022-05-09'
until_date = '2024-04-22'

spy_intra_data = fetch_polygon_data(ticker, from_date, until_date, 'minute')
spy_daily_data = fetch_polygon_data(ticker, from_date, until_date, 'day')
dividends      = fetch_polygon_dividends(ticker)

'''
