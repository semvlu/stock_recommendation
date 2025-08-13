# Stock Recommendation System

A rule-based + LLM stock recommendation system

Market Support: AMS, PAR, NASDAQ

## Methodology

- CBOE Volatility Index (^VIX) for market volatility measurement, i.e. weight for chaos score.
- User custom chaos score.
- Compare company EV/EBITDA, P/E, P/B ratios within the sector to obtain scores in finance.
- News source filtering for high quality RAG input, e.g. Bloomberg, Politico, Reuters, Financial Times, Investopedia.
- Linear regression as preliminary price prediction.
- PoC: stock price 5 days onward.

Final score: $$\sigma = \sum\_{i=1}^{4} w_i s_i$$,

where $w$ stands for weight and $s$ stands for scores. There are 4 scores, i.e.

- News
- Chaos
- Financial
- Index

## Models

Llama 3.1 8B

Sentence Transformers: all-MiniLM-L6-v2

## RAG

FAISS: IndexFlatL2

## Tools

Streamlit, yfinance, NewsAPI, Beautiful Soup, Altair, RegEx, NLTK

Source:
[snp500.csv](https://datahub.io/core/s-and-p-500-companies)
[Euronext_Equities_XAMS.csv](https://live.euronext.com/en/markets/amsterdam/equities/list)
[Euronext_Equities_XPAR.csv](https://live.euronext.com/en/markets/paris/equities/list)

![alttext](https://github.com/semvlu/stock_recommendation/blob/main/preview.png?raw=true)
