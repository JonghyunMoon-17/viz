import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import linregress, mannwhitneyu

st.set_page_config(layout="wide", page_title="가격 상승률 vs 거래 변화 시각화")

# =========================
# 0) 데이터 로드
# =========================
df = pd.read_csv("district_summary.csv").copy()
df = df[df["trade_count_2023"] > 0].copy()

# 절대지표용: 거래건수 증가율
df["trade_count_growth"] = (df["trade_count_2025"] - df["trade_count_2023"]) / df["trade_count_2023"]

# =========================
# 1) 급지(1~4) 매핑 (팀 기준대로 여기만 수정)
# =========================
GRADE_MAP = {
    # 예시(임시). 팀 기준으로 바꿔 쓰면 됨.
    "강남구": 1, "서초구": 1, "송파구": 1, "용산구": 1,
    "성동구": 2, "마포구": 2, "강동구": 2, "광진구": 2,
    "동작구": 3, "서대문구": 3, "영등포구": 3, "동대문구": 3,
    "금천구": 4, "강북구": 4, "도봉구": 4, "노원구": 4,
}
df["급지"] = df["구명"].map(GRADE_MAP).fillna(0).astype(int)  # 0 = 미분류

# =========================
# 2) UI 상단
# =========================
st.title("가격 상승률 vs 거래 변화 (통합 시각화)")

mode = st.radio(
    "지표 모드 선택",
    ["절대지표(거래건수)", "상대지표(거래비중)"],
    horizontal=True
)

left, right = st.columns([2.2, 1])

# =========================
# 3) 급지 선택 = 필터링 X, 하이라이트 O
# =========================
with right:
    st.markdown("### 통계")

    grade_opt = st.radio(
        "급지 강조(1~4급지)",
        ["전체", "1급지", "2급지", "3급지", "4급지"],
        horizontal=True
    )

# 모든 구는 유지하되, 선택 급지만 빨간색으로 강조
df_plot = df.copy()
if grade_opt == "전체":
    df_plot["highlight"] = "전체"
else:
    g = int(grade_opt.replace("급지", ""))
    df_plot["highlight"] = np.where(df_plot["급지"] == g, f"{g}급지", "기타")

# =========================
# 4) 산점도 설정 (x 고정, y는 모드별)
# =========================
x_col = "price_growth"
if mode == "절대지표(거래건수)":
    y_col = "trade_count_growth"
    y_label = "거래건수 증가율 (2023→2025)"
else:
    y_col = "trade_share_change"
    y_label = "거래 비중 변화 (2023→2025)"

# =========================
# ✅ (핵심) 버블 사이즈를 app2.py 스타일로 “정규화 스케일”로 고정
# - raw 거래건수(또는 sqrt)는 구간이 너무 넓어서 시각적으로 불편해짐
# - app2에서 흔히 쓰는 방식 = min/max 정규화 후 일정 범위로 매핑
# =========================
size_raw = df_plot["trade_count_2025"].astype(float).clip(lower=1)

# 상위 극단 때문에 크게 튀는 걸 막기 위해 winsorize(95% 상한 클리핑)
cap = np.quantile(size_raw, 0.95)
size_w = np.minimum(size_raw, cap)

# 0~1 정규화
size_norm = (size_w - size_w.min()) / (size_w.max() - size_w.min() + 1e-9)

# 화면에서 예쁘게 보이는 범위로 매핑 (app2 느낌: 과하지 않게)
# 필요하면 여기만 조금 조절하면 됨.
df_plot["bubble_size"] = 8 + size_norm * 22   # 8~30 정도

# =========================
# 라벨(구 이름) 최소화: 강남3구 + x상위3 + y상/하위3 (기존 방식 유지)
# =========================
core = ["강남구", "서초구", "송파구"]
top_x = df_plot.nlargest(3, x_col)["구명"].tolist()
top_y = df_plot.nlargest(3, y_col)["구명"].tolist()
bot_y = df_plot.nsmallest(3, y_col)["구명"].tolist()
label_set = set(core + top_x + top_y + bot_y)
df_plot["label"] = df_plot["구명"].where(df_plot["구명"].isin(label_set), "")

# =========================
# 회귀 통계
# =========================
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

# =========================
# Hover(툴팁) — 한국어 + 순서 + label 숨김
# =========================
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
        "label": False,
        "bubble_size": False,
        "highlight": False,
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
        "label": False,
        "bubble_size": False,
        "highlight": False,
    }

# =========================
# 5) 좌측: 산점도 렌더
# =========================
with left:
    st.subheader("산점도")

    # 색상: 선택 급지 = 빨강, 나머지 = 기존 파란계열(연하게)
    if grade_opt == "전체":
        color_map = {"전체": "#4C78A8"}  # 파랑
    else:
        g = int(grade_opt.replace("급지", ""))
        color_map = {f"{g}급지": "#E45756", "기타": "#4C78A8"}  # 빨강/파랑

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        size="bubble_size",
        size_max=30,  # ✅ 정규화된 size를 쓰므로 max도 작게 고정 가능
        color="highlight",
        color_discrete_map=color_map,
        hover_name="구명",
        text="label",
        trendline="ols",
        labels={
            x_col: "가격 상승률 (2023→2025)",
            y_col: y_label,
            "highlight": "급지 강조"
        },
        hover_data=hover_cols
    )
    fig.update_traces(textposition="top center", marker=dict(opacity=0.85))

    # 범례가 너무 거슬리면 숨겨도 됨(원하면 True->False)
    fig.update_layout(legend_title_text="")

    st.plotly_chart(fig, use_container_width=True)

# =========================
# 6) 추가분석: 조건부 비교 (좌/우) + ✅ q 슬라이더 추가
# =========================
st.divider()
st.header("추가분석: 가격 ↔ 거래량 선행 가능성(분석 A/B)")
st.caption("‘지표 변동이 큰 구 vs 작은 구’로 나눠서 다른 지표의 분포를 비교")

# ✅ 분위수 q 슬라이더 (상위 q / 하위 (1-q))
# 예: q=0.7이면 상위 30%, 하위 30%를 비교
q = st.slider("상/하위 그룹 분리 분위수 q (상위 q, 하위 1-q)", 0.60, 0.90, 0.70, 0.01)

col1, col2 = st.columns(2)

# -------- 분석 1: 거래량 변동 상/하위 → 가격 상승률 --------
with col1:
    st.subheader("분석 A: 거래량 변동 상/하위 → 가격 상승률")

    q_hi = df["trade_count_growth"].quantile(q)
    q_lo = df["trade_count_growth"].quantile(1 - q)

    B_high = df[df["trade_count_growth"] >= q_hi].copy()
    B_low  = df[df["trade_count_growth"] <= q_lo].copy()

    B_high["group"] = f"거래량 변동 상위(≥{q:.2f})"
    B_low["group"]  = f"거래량 변동 하위(≤{1-q:.2f})"
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

# -------- 분석 2: 가격 변동 상/하위 → 거래량 증가율 --------
with col2:
    st.subheader("분석 B: 가격 변동 상/하위 → 거래량 증가율")

    p_hi = df["price_growth"].quantile(q)
    p_lo = df["price_growth"].quantile(1 - q)

    A_high = df[df["price_growth"] >= p_hi].copy()
    A_low  = df[df["price_growth"] <= p_lo].copy()

    A_high["group"] = f"가격 변동 상위(≥{q:.2f})"
    A_low["group"]  = f"가격 변동 하위(≤{1-q:.2f})"
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
