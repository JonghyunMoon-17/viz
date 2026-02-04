import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import linregress, mannwhitneyu

st.set_page_config(layout="wide", page_title="가격 상승률 vs 거래 변화 (통합 시각화)")

# =========================
# 0) 데이터 로드
# =========================
df = pd.read_csv("district_summary.csv").copy()

# 안전장치: 거래건수 0인 경우 분모 이슈 방지
df = df[df["trade_count_2023"] > 0].copy()

# 절대지표용: 거래건수 증가율
df["trade_count_growth"] = (df["trade_count_2025"] - df["trade_count_2023"]) / df["trade_count_2023"]

# =========================
# 1) 급지(1~4) 매핑 (팀 기준대로 여기만 수정)
# =========================
# ⚠️ 너희 팀에서 정의한 1~4급지 기준이 있으면 여기 dict만 바꾸면 됨.
GRADE_MAP = {
    # 1급지 예시(임시)
    "강남구": 1, "서초구": 1, "송파구": 1, "용산구": 1,
    # 2급지 예시(임시)
    "성동구": 2, "마포구": 2, "강동구": 2, "광진구": 2,
    # 3급지 예시(임시)
    "동작구": 3, "서대문구": 3, "영등포구": 3, "동대문구": 3,
    # 4급지 예시(임시)
    "금천구": 4, "강북구": 4, "도봉구": 4, "노원구": 4,
}

df["급지"] = df["구명"].map(GRADE_MAP).fillna(0).astype(int)  # 0 = 미분류

# =========================
# 2) UI 상단
# =========================
st.title("가격 상승률 vs 거래 변화 (통합 시각화)")
st.caption("절대지표(거래건수)와 상대지표(거래비중)를 버튼 하나로 전환하며, 원 크기는 2025 거래건수 기반으로 동일하게 유지합니다.")

mode = st.radio(
    "지표 모드 선택",
    ["절대지표(거래건수)", "상대지표(거래비중)"],
    horizontal=True
)

# 우측 패널 아래에 급지 필터를 두고 싶어서, 전체 레이아웃을 2열로
left, right = st.columns([2.2, 1])

# =========================
# 3) 급지 필터 (우측 패널 하단에 위치)
# =========================
with right:
    st.markdown("### 해석 요약(통계)")
    st.caption("선택한 지표 모드(절대/상대)에 따라 y축 및 회귀 결과가 자동 업데이트됩니다.")

    grade_opt = st.radio(
        "급지 강조(1~4급지)",
        ["전체", "1급지", "2급지", "3급지", "4급지"],
        horizontal=True
    )

# 필터 적용
df_plot = df.copy()
if grade_opt != "전체":
    g = int(grade_opt.replace("급지", ""))
    df_plot = df_plot[df_plot["급지"] == g].copy()

# =========================
# 4) 산점도: x 고정(가격상승률), y는 모드에 따라
# =========================
x_col = "price_growth"
if mode == "절대지표(거래건수)":
    y_col = "trade_count_growth"
    y_label = "거래건수 증가율 (2023→2025)"
else:
    y_col = "trade_share_change"
    y_label = "거래 비중 변화 (2023→2025)"

# 원 크기(2025 거래건수) 스케일링: 너무 커지는 문제 방지
# - sqrt 스케일 + size_max로 ‘기존 상대지표 페이지 느낌’ 유지
df_plot["bubble_size"] = np.sqrt(df_plot["trade_count_2025"].clip(lower=1))

# 라벨(구 이름) 최소화: 강남3구 + x상위3 + y상/하위3
core = ["강남구", "서초구", "송파구"]
top_x = df_plot.nlargest(3, x_col)["구명"].tolist()
top_y = df_plot.nlargest(3, y_col)["구명"].tolist()
bot_y = df_plot.nsmallest(3, y_col)["구명"].tolist()
label_set = set(core + top_x + top_y + bot_y)
df_plot["label"] = df_plot["구명"].where(df_plot["구명"].isin(label_set), "")

# 회귀 통계(상관/기울기/R2/p)
# - 결측/상수 방어
tmp = df_plot[[x_col, y_col]].dropna()
if len(tmp) >= 3 and tmp[x_col].nunique() > 1:
    lr = linregress(tmp[x_col], tmp[y_col])
    r = lr.rvalue
    beta = lr.slope
    r2 = lr.rvalue ** 2
    p_beta = lr.pvalue
else:
    r = beta = r2 = p_beta = np.nan

with right:
    st.markdown(
        f"""
- **상관계수 r:** `{r:.3f}`  
- **회귀 기울기 β:** `{beta:.4f}`  
- **R²:** `{r2:.3f}`  
- **p-value (β):** `{p_beta:.4f}`
        """
    )

# Hover(툴팁) — label 같은 내부값 제거, 한국어 + 순서 정리
if mode == "절대지표(거래건수)":
    hover_cols = {
        "구명": True,
        "median_price_2023": ":,.0f",
        "median_price_2025": ":,.0f",
        "price_growth": ":.3f",
        "trade_count_2023": ":,",
        "trade_count_2025": ":,",
        "trade_count_growth": ":.2%",
        "trade_share_2023": ":.3f",
        "trade_share_2025": ":.3f",
        "trade_share_change": ":.3f",
        "급지": True,
        "bubble_size": False,
        "label": False,
    }
else:
    hover_cols = {
        "구명": True,
        "median_price_2023": ":,.0f",
        "median_price_2025": ":,.0f",
        "price_growth": ":.3f",
        "trade_share_2023": ":.3f",
        "trade_share_2025": ":.3f",
        "trade_share_change": ":.3f",
        "trade_count_2023": ":,",
        "trade_count_2025": ":,",
        "trade_count_growth": ":.2%",
        "급지": True,
        "bubble_size": False,
        "label": False,
    }

# =========================
# 5) 좌측: 산점도 렌더
# =========================
with left:
    st.subheader("산점도")

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        size="bubble_size",
        size_max=42,  # ✅ 핵심: 원이 과하게 커지지 않게 제한
        hover_name="구명",
        text="label",
        trendline="ols",
        labels={
            x_col: "가격 상승률 (2023→2025)",
            y_col: y_label,
        },
        hover_data=hover_cols
    )
    fig.update_traces(textposition="top center")

    st.plotly_chart(fig, use_container_width=True)

# =========================
# 6) 추가분석: 조건부 비교(분석 1/2) — 좌/우 배치, 박스플롯 유지, slope 제거
# =========================
st.divider()
st.header("추가분석: 가격 ↔ 거래량 선행 가능성(조건부 비교, 분석 1/2)")
st.caption("‘한쪽 변동이 큰 구 vs 작은 구’로 나눠서, 다른 지표의 반응(분포)을 비교합니다. (비모수 검정: Mann–Whitney U)")

col1, col2 = st.columns(2)

# -------- 분석 1 (덜 잘 나온 것 먼저): 거래량 변동 상/하위 → 가격상승률 비교 --------
with col1:
    st.subheader("분석 1: 거래량 변동 상/하위 → 가격 상승률")
    q_hi, q_lo = df["trade_count_growth"].quantile(0.7), df["trade_count_growth"].quantile(0.3)

    B_high = df[df["trade_count_growth"] >= q_hi].copy()
    B_low  = df[df["trade_count_growth"] <= q_lo].copy()

    B_high["group"] = "거래량 변동 상위"
    B_low["group"]  = "거래량 변동 하위"
    B_df = pd.concat([B_high, B_low], ignore_index=True)

    figB = px.box(
        B_df,
        x="group",
        y="price_growth",
        points="all",
        labels={"group": "구분", "price_growth": "가격 상승률 (2023→2025)"},
        title="(Boxplot) 거래량 변동 그룹별 가격 상승률"
    )
    st.plotly_chart(figB, use_container_width=True)

    if len(B_high) > 0 and len(B_low) > 0:
        uB = mannwhitneyu(B_high["price_growth"], B_low["price_growth"], alternative="two-sided")
        st.write(f"- **p-value:** `{uB.pvalue:.4f}`")

# -------- 분석 2 (잘 나온 것 나중): 가격 변동 상/하위 → 거래량 증가율 비교 --------
with col2:
    st.subheader("분석 2: 가격 변동 상/하위 → 거래량 증가율")
    p_hi, p_lo = df["price_growth"].quantile(0.7), df["price_growth"].quantile(0.3)

    A_high = df[df["price_growth"] >= p_hi].copy()
    A_low  = df[df["price_growth"] <= p_lo].copy()

    A_high["group"] = "가격 변동 상위"
    A_low["group"]  = "가격 변동 하위"
    A_df = pd.concat([A_high, A_low], ignore_index=True)

    figA = px.box(
        A_df,
        x="group",
        y="trade_count_growth",
        points="all",
        labels={"group": "구분", "trade_count_growth": "거래건수 증가율 (2023→2025)"},
        title="(Boxplot) 가격 변동 그룹별 거래량 증가율"
    )
    st.plotly_chart(figA, use_container_width=True)

    if len(A_high) > 0 and len(A_low) > 0:
        uA = mannwhitneyu(A_high["trade_count_growth"], A_low["trade_count_growth"], alternative="two-sided")
        st.write(f"- **p-value:** `{uA.pvalue:.4f}`")
