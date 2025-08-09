import streamlit as st
import altair as alt
import yfinance as yf
import numpy as np
import random
import re
import pandas as pd
from linear_regression_model import linRegVis, get_selectors_chart, get_points_chart
from llmAPI import get_response
from score import capm, wacc, capm_wacc_score, get_financial_scores, get_final_score

st.title("Stock Investment Recommendation")

main, chat = st.columns([3, 2])

with main:
    @st.cache_data
    def load_symbols():
        return pd.read_csv('src/symbol.csv')
    df = load_symbols()

    with st.form("search"):
        query = st.text_input(
            "Type to search for a stock (symbol or name):",
            key="query",
            placeholder="e.g., AAPL, TSLA, etc."
        )

        if query:
            filtered = df[
                df['Symbol'].str.contains(query, case=False, na=False) |
                df['Name'].str.contains(query, case=False, na=False)
            ]
        else:
            filtered = df

        stocks = filtered["Symbol"] + " - " + filtered["Name"]

        choice = st.selectbox(
            "Choose a stock from filtered results:",
            options=stocks if not stocks.empty else ["No match found"]
        )

        chaos = st.slider(
        "Chaos (Order vs. Chaos)", 
        0, 10, 1,
        help="Higher: more unpredictable influence"
        )

        # Chaos level
        if chaos == 0.0:
            random_score = 0.0
        else:
            random_score = max(-1.0, min(1.0,random.gauss(mu=0, sigma=chaos/10)))

        submit = st.form_submit_button("Search🤑")



    def clean_company_name(name):
        # Split at the first occurrence of any of the keywords and keep only the part before
        parts = re.split(
            r'\b(Inc\.?|Incorporated|Ltd\.?|Limited|Corp\.?|Corporation|LLC|PLC|S\.A\.|AG|N\.V\.|Co\.?|Common [A-Z]* Stock|Class [A-Z]|Ordinary Shares)\b\.?',
            name, flags=re.IGNORECASE, maxsplit=1
        )
        return parts[0].strip(" ,")
    # Visualisation
    if choice != "No match found":
        symbol = choice.split(" - ")[0].strip()
        name = choice.split(" - ", 1)[1].strip()
        name = clean_company_name(name)

    dat = yf.Ticker(symbol)
    hist = dat.history(period='5d', interval='1h').reset_index()
    hist['Datetime'] = pd.to_datetime(hist['Datetime'])  # Ensure correct dtype





    plot_data, model, initial_price = linRegVis(hist)
    nearest = alt.selection_point(nearest=True, on="mouseover", fields=['Datetime'], empty="none")

    line = alt.Chart(plot_data).mark_line().encode(
        x='Datetime:T',
        y=alt.Y('Close:Q', scale=alt.Scale(domain=[plot_data['Close'].min(), plot_data['Close'].max()], nice=True)),
        color='Type:N',
        tooltip=[
            alt.Tooltip('Datetime:T', title='Date'),
            alt.Tooltip('Close:Q', title='Price'),
            alt.Tooltip('Type:N')
        ]
    )
    selectors = get_selectors_chart(plot_data, nearest)
    points = get_points_chart(line, nearest)
    rules = alt.Chart(plot_data).mark_rule(color='gray').encode(
        x='Datetime:T',
        opacity=alt.condition(nearest, alt.value(0.5), alt.value(0)),
        tooltip=[
            alt.Tooltip('Datetime:T', title='Date'),
            alt.Tooltip('Close:Q', title='Price'),
            alt.Tooltip('Type:N')
        ]
    ).add_params(
        nearest
    )

    linRegPred = alt.layer(
        line, selectors, points, rules
    ).properties(
        title = symbol + " Price with Regression Trend Line (5d, 1h)"
    ).interactive()


    st.altair_chart(linRegPred, use_container_width=True)

    # Predict price 5 days (120 hours) into the future
    future = (hist['Datetime'] - hist['Datetime'].iloc[0]).dt.total_seconds().max() / 3600.0 + 120
    pred_log = model.predict([[future]])[0]
    pred_actual = hist['Close'].iloc[0] * np.exp(pred_log)

    # No need to wait for LLM resp & financial scores
    if model.coef_[0] >= 0:
        st.subheader(f"📈 Model 5-Day Forward Price Prediction: **${pred_actual:.2f}**")
    else:
        st.subheader(f"📉 Model 5-Day Forward Price Prediction: **${pred_actual:.2f}**")
    st.write("This prediction is based on a linear regression model fitted to the last 5 days of hourly data. Without considering external factors.")
    # ----------------------------


    if submit:
        capm = capm(dat)
        wacc = wacc(dat, capm)
        capm_wacc_score = capm_wacc_score(capm, wacc)
        financial_scores = get_financial_scores(symbol)
        financial_scores.update({
            "capm_wacc": capm_wacc_score
        })
        fin_score = 0.25 * sum(financial_scores.values())

        st.subheader(f"CAPM: {capm:.4f}")
        st.subheader(f"WACC: {wacc:.4f}")
        st.subheader(f"EV/EBITDA: {financial_scores['ev_ebitda']:.4f}")
        st.subheader(f"P/E: {financial_scores['pe_ratio']:.4f}")
        st.subheader(f"P/B: {financial_scores['pb_ratio']:.4f}")

        slope = model.coef_[0]
        index_score = np.tanh(slope / 2)


        initial_msg = get_response(_q=name, query=f"What are the latest news and outlook for {name}?")
        print(initial_msg)
        match = re.search(r'\*{0,2}score\*{0,2}\s*:\s*([-+]?\d*\.?\d+)', initial_msg, re.IGNORECASE)        
        news_score = float(match.group(1))
        subscores = {
            "fin": fin_score,
            "news": news_score,
            "index": index_score,
            "random": random_score
        }
        final_score = get_final_score(subscores)
        if final_score > 0.3:
            decision = "BUY"
        elif final_score < -0.3:
            decision = "SELL"
        else:
            decision = "HOLD"
        st.subheader(f"Final Score: {final_score:.2f}, Decision: {decision}")



with chat:
    # Init chat history in session_state if not present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Init LLM message
    if not st.session_state.get("llm_greeted", False):
        st.session_state["chat_history"].append({"role": "assistant", "content": initial_msg})
        st.session_state["llm_greeted"] = True

    # Display all messages in order
    messages = st.container(height=500)
    for msg in st.session_state["chat_history"]:
        messages.chat_message(msg["role"]).write(msg["content"])

    # Place chat input AFTER the messages
    prompt = st.chat_input("Ask some questions")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        response = get_response(_q=name, query=prompt)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})


# Run: streamlit run app.py