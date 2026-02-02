import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

st.set_page_config(layout="wide")

st.title("결론: 가격 상승률 vs 거래 비중 변화")
st.caption("2023–2025 서울 25개 구 | 가격 상승 지역에 수요(거래 비중)가 추가로 집중되는지 검증")

df = pd.read_csv("district_summary.csv")

X = sm.add_constant(df["price_growth"])
y = df["trade_share_change"]
model = sm.OLS(y, X).fit()

p_value = model.pvalues["price_growth"]
slope = model.params["price_growth"]
r2 = model.rsquared

# 안전장치: 컬럼 확인
required = {"구명", "price_growth", "trade_share_change"}
if not required.issubset(df.columns):
    st.error(f"district_summary.csv 컬럼이 부족합니다. 필요한 컬럼: {required}, 현재: {set(df.columns)}")
    st.stop()

# 상관계수
corr = df["price_growth"].corr(df["trade_share_change"])

# 라벨링할 핵심 구(원하면 추가/삭제)
highlight = ["강남구", "서초구", "송파구", "동작구", "성동구", "마포구", "광진구", "용산구", "강동구"]
df["label"] = df["구명"].where(df["구명"].isin(highlight), "")

# 버블 크기 (없으면 1로 대체)
if "trade_count_2025" in df.columns:
    size_col = "trade_count_2025"
else:
    df["trade_count_2025"] = 1
    size_col = "trade_count_2025"

# -------- Layout --------
left, right = st.columns([2.2, 1])

with left:
    st.subheader("산점도: 가격 상승률(X) vs 거래 비중 변화(Y)")

    # 메인 산점도 (툴팁 강화)
    fig = px.scatter(
        df,
        x="price_growth",
        y="trade_share_change",
        size="trade_count_2025",
        hover_name="구명",
        text="label",
        trendline="ols",
        hover_data={
        "median_price_2023": ":,.0f",
        "median_price_2025": ":,.0f",
        "price_growth": ":.3f",
        "trade_share_2023": ":.3f",
        "trade_share_2025": ":.3f",
        "trade_share_change": ":.3f",
        "trade_count_2023": True,
        "trade_count_2025": True,
        },
        labels={
        "price_growth": "가격 상승률 (2023→2025)",
        "trade_share_change": "거래 비중 변화 (2023→2025)",
        "median_price_2023": "2023 중위가격",
        "median_price_2025": "2025 중위가격",
        "trade_share_2023": "2023 거래비중",
        "trade_share_2025": "2025 거래비중",
        "trade_count_2023": "2023 거래건수",
        "trade_count_2025": "2025 거래건수"
        },
    )

    # 사분면 기준선 (x=0, y=0)
    fig.add_hline(y=0, line_width=1, line_dash="dash")
    fig.add_vline(x=0, line_width=1, line_dash="dash")

    # 보기 좋게
    fig.update_traces(textposition="top center")
    fig.update_layout(height=620)

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("해석 요약")

    st.markdown(
        f"""
**상관계수:** `{corr:.2f}`  

**사분면 해석(핵심):**
- **Q1 (X+, Y+)**: 가격 상승, 거래비중 상승 → 되먹임 구조(가설 지지)
- **Q4 (X+, Y-)**: 가격 상승, 거래비중 하락 → 기저효과/희석 가능성(예: 강남구)
- **Q2 (X-, Y+)**: 가격 약세/정체 거래비중 상승 → 대체 수요 가능성
- **Q3 (X-, Y-)**: 약세 지역
        """
    )

    st.info(
        "강남구의 비중 감소는 수요 이탈이라기보다, 전체 거래량 확대 속 상대적 비중이 희석된 결과일 수 있음."
        "       "
        "핵심은 Q1에 위치한 ‘가격 상승 + 거래비중 상승’ 지역이 존재하는지와 그 분포로, Q1에 ‘동작/성동/마포/광진’ 같은 준고가 축이 존재한다는 점, 이는 가격 신호가 수요를 억제하지 못하는 ‘되먹임 구조’를 시사."
    )

    # Top/Bottom tables
    st.markdown(
        f"""
    **상관계수:** `{corr:.2f}`  
    **회귀 기울기(β):** `{slope:.3f}`  
    **R²:** `{r2:.2f}`  
    **p-value (β):** `{p_value:.4f}`  
    """
    )    

    st.markdown("### 가격 상승률 Top 5")
    top_price = df.sort_values("price_growth", ascending=False).head(5)[["구명", "price_growth"]]
    st.dataframe(top_price, use_container_width=True)

    st.markdown("### 거래 비중 증가 Top 5")
    top_share = df.sort_values("trade_share_change", ascending=False).head(5)[["구명", "trade_share_change"]]
    st.dataframe(top_share, use_container_width=True)

    st.markdown("### Q1 (X+, Y+)")
    q1 = df[(df["price_growth"] > 0) & (df["trade_share_change"] > 0)][["구명", "price_growth", "trade_share_change"]]
    q1 = q1.sort_values(["trade_share_change", "price_growth"], ascending=False)
    st.dataframe(q1, use_container_width=True)

    st.markdown("### Q4 (X+, Y-)")
    q1 = df[(df["price_growth"] > 0) & (df["trade_share_change"] < 0)][["구명", "price_growth", "trade_share_change"]]
    q1 = q1.sort_values(["trade_share_change", "price_growth"], ascending=False)
    st.dataframe(q1, use_container_width=True)
