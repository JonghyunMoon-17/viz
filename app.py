import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide")

df = pd.read_csv("district_summary.csv")
df = df[df["trade_count_2023"] > 0].copy()

# --- (A) 거래건수 증가율 컬럼 생성 ---
df["trade_count_growth"] = (df["trade_count_2025"] - df["trade_count_2023"]) / df["trade_count_2023"]

# --- (B) 상관계수 ---
corr2 = df["price_growth"].corr(df["trade_count_growth"])

# --- (C) 라벨 최소화: 강남3구 + 거래증가율 Top3/Bottom3 + 가격상승 Top3 ---
top_price = df.nlargest(3, "price_growth")["구명"].tolist()
top_tc = df.nlargest(3, "trade_count_growth")["구명"].tolist()
bot_tc = df.nsmallest(3, "trade_count_growth")["구명"].tolist()
core = ["강남구", "서초구", "송파구"]

label_set = set(top_price + top_tc + bot_tc + core)
df["label_tc"] = df["구명"].where(df["구명"].isin(label_set), "")

# --- (D) Plotly 산점도 + 회귀선 ---
fig2 = px.scatter(
    df,
    x="price_growth",
    y="trade_count_growth",
    hover_name="구명",
    text="label_tc",
    trendline="ols",
    labels={
        "price_growth": "가격 상승률 (2023→2025)",
        "trade_count_growth": "거래건수 증가율 (2023→2025)",
    },
    hover_data={
        "trade_count_2023": True,
        "trade_count_2025": True,
        "price_growth": ":.3f",
        "trade_count_growth": ":.2%",
    },
)

# 텍스트 라벨 위치 살짝 조정
fig2.update_traces(textposition="top center")

# --- (E) Streamlit에 출력 ---
st.subheader("가격 상승률 vs 거래건수 증가율")
st.caption(f"거래 비중(상대지표)의 한계를 보완하기 위해 절대 거래량(거래건수) 기준의 관계를 추가 확인 | 상관계수 r = {corr2:.2f}")
st.plotly_chart(fig2, use_container_width=True)

# Q1 / Q4 기준선 (0 기준)
df["group"] = "Other"

df.loc[
    (df["price_growth"] > 0) & (df["trade_share_change"] > 0),
    "group"
] = "Q1 (가격↑ · 비중↑)"

df.loc[
    (df["price_growth"] > 0) & (df["trade_share_change"] < 0),
    "group"
] = "Q4 (가격↑ · 비중↓)"

# 분석 대상은 Q1, Q4만
df_box = df[df["group"].isin(["Q1 (가격↑ · 비중↑)", "Q4 (가격↑ · 비중↓)"])].copy()

df_box = df_box[df_box["trade_count_2023"] > 0].copy()
df_box["trade_count_growth"] = (
    df_box["trade_count_2025"] - df_box["trade_count_2023"]
) / df_box["trade_count_2023"]

import plotly.express as px

fig_box = px.box(
    df_box,
    x="group",
    y="trade_count_growth",
    points="all",  # 점도 같이 보여줌 (설득력 ↑)
    labels={
        "group": "구분",
        "trade_count_growth": "거래건수 증가율 (2023→2025)",
    },
)

st.subheader("Q1 vs Q4 거래건수 증가율 비교")
st.caption(
    "가격 상승률은 유사하지만, 거래 비중 방향에 따라 절대 거래량의 분포가 어떻게 달라지는지 비교"
)
st.plotly_chart(fig_box, use_container_width=True)