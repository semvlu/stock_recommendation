import random
import yfinance as yf
import pandas as pd

# CBOE Volatility Index (^VIX)
vix = yf.Ticker("^VIX").history(period="1d")
vix_value = vix['Close'].iloc[-1] if not vix.empty else 0

weights = {
    "fin": 0.45,
    "news": 0.35,
    "index": 0.2,
    "random": 0.15 if vix_value > 30 else 0.1 if vix_value > 20 else 0.05
}
ttl = sum(weights.values())

def capm(dat):
    '''
    CAPM: Capital Asset Pricing Model
    ri: Expected return of investment
    rf: Risk-free rate (e.g., yield on 10-year Treasury bond)
    rm: Expected return of the market
    '''
    tnx = yf.Ticker("^TNX")
    Rf = tnx.info['regularMarketPrice'] / 100

    sp500 = yf.Ticker("^GSPC")  # S&P 500 index
    hist = sp500.history(period="10y")
    start_price = hist['Close'].iloc[0]
    end_price = hist['Close'].iloc[-1]
    Rm = ((end_price / start_price) ** (1 / 10)) - 1

    Ri = Rf + dat.info['beta'] * (Rm - Rf)
    return Ri

def wacc(dat, Ri):
    '''
    WACC: Weighted Average Cost of Capital
    '''

    E = dat.info['marketCap']
    D = dat.info['totalDebt']
    Re = Ri

    ttlDebt = dat.info['totalDebt']
    # Get the most recent non-NaN value
    interestExpense = dat.financials.loc['Interest Expense'][~dat.financials.loc['Interest Expense'].isna()].iloc[0]
    interestExpense = abs(interestExpense)
    Rd = interestExpense / ttlDebt

    T = 0.21 # Corporate tax rate, assuming 21% for US companies
    V = E + D
    WACC = (E / V) * Re + (D / V) * Rd * (1 - T)
    return WACC

def capm_wacc_score(capm, wacc):
    # Normalize to between -1 and +1
    score = max(-1, min(1, (capm - wacc) / 5))  # assumes +/-5% extremes
    return score

# Financial scores: EV/EBITDA, P/E, P/B
df = pd.read_csv('src/snp500.csv')

def get_financial_scores(symbol):
    dat = yf.Ticker(symbol)
    enterprise_value = dat.info['enterpriseValue']
    ebidta = dat.info['ebitda']
    ev_ebitda = enterprise_value / ebidta
    pe_ratio = dat.info["trailingPE"]
    pb_ratio = dat.info["priceToBook"]

    sector = df[df['Symbol'] == symbol]['GICS Sector'].values[0] if not df[df['Symbol'] == symbol].empty else "Unknown"

    competitors = df[df['GICS Sector'] == sector]['Symbol'].tolist()
    competitors.remove(symbol)  # Remove the current symbol from its competitors

    avg = {
        "ev_ebitda": 0.0,
        "pe_ratio": 0.0,
        "pb_ratio": 0.0
    }
    for i in competitors:
        dat_comp = yf.Ticker(i)
        try:
            avg["ev_ebitda"] += dat_comp.info['enterpriseValue'] / dat_comp.info['ebitda']
            avg["pe_ratio"] += dat_comp.info['trailingPE']
            avg["pb_ratio"] += dat_comp.info['priceToBook']
        except KeyError:
            print(f"Financial data for {i} is not available")
            
    for key in avg:
        if len(competitors) > 0:
            avg[key] /= len(competitors)
        

    ev_ebitda_score = avg['ev_ebitda'] / ev_ebitda 
    pe_ratio_score = avg['pe_ratio'] / pe_ratio
    pb_ratio_score = avg['pb_ratio'] / pb_ratio
    return {
        "ev_ebitda": ev_ebitda_score,
        "pe_ratio": pe_ratio_score,
        "pb_ratio": pb_ratio_score
    }



def get_final_score(subscores):
    score = 0.0
    for key in weights:
        weights[key] /= ttl
        score += weights[key] * subscores[key]
    return score


# PoC: prediction on 5 days later