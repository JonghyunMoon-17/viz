import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import mannwhitneyu

st.set_page_config(layout="wide")

# -----------------------------
# Data Load
# -----------------------------
@st.cache_data
def load_summary(path: str = "district_summary.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    # 실험에 필요한 최소 컬럼은 결측 제거
    need = ["구명", "price_growth", "trade_share_change", "trade_count_2023", "trade_count_2025"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"district_summary.csv에 '{c}' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")
    # 2023 거래건수가 0이면 증가율 계산이 깨지므로 제외
    df = df[df["trade_count_2023"] > 0].copy()
    return df

df = load_summary("district_summary.csv")

# --- (A) 거래건수 증가율 컬럼 생성 (이미 있더라도 안전하게 덮어씀)
df["trade_count_growth"] = (df["trade_count_2025"] - df["trade_count_2023"]) / df["trade_count_2023"]

# -----------------------------
# 1) Scatter (절대지표)
# -----------------------------
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
        "median_price_2023": True if "median_price_2023" in df.columns else False,
        "median_price_2025": True if "median_price_2025" in df.columns else False,
        "trade_share_2023": True if "trade_share_2023" in df.columns else False,
        "trade_share_2025": True if "trade_share_2025" in df.columns else False,
        "trade_share_change": ":.3f",
        "price_growth": ":.3f",
        "trade_count_growth": ":.2%",
    },
)

fig2.update_traces(textposition="top center")

st.subheader("가격 상승률 vs 거래건수 증가율(절대지표)")
st.caption(
    f"거래 비중(상대지표)의 한계를 보완하기 위해 절대 거래량(거래건수) 기준의 관계를 추가 확인 | 상관계수 r = {corr2:.2f}"
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# -----------------------------
# 2) Q1 vs Q4 비교 (기존 유지)
# -----------------------------
df_q = df.copy()
df_q["group"] = "Other"

df_q.loc[(df_q["price_growth"] > 0) & (df_q["trade_share_change"] > 0), "group"] = "Q1 (가격↑ · 비중↑)"
df_q.loc[(df_q["price_growth"] > 0) & (df_q["trade_share_change"] < 0), "group"] = "Q4 (가격↑ · 비중↓)"

df_box = df_q[df_q["group"].isin(["Q1 (가격↑ · 비중↑)", "Q4 (가격↑ · 비중↓)"])].copy()

fig_box = px.box(
    df_box,
    x="group",
    y="trade_count_growth",
    points="all",
    labels={
        "group": "구분",
        "trade_count_growth": "거래건수 증가율 (2023→2025)",
    },
    title="Q1 vs Q4 거래건수 증가율 비교"
)

st.subheader("Q1 vs Q4 거래건수 증가율 비교")
st.caption("가격 상승률은 유사하지만, 거래 비중 방향에 따라 절대 거래량의 분포가 어떻게 달라지는지 비교")
st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# -----------------------------
# 3) 추가분석: Lead–Lag (실험 A/B) + Boxplot + Slope chart + p-value
# -----------------------------
with st.expander("추가분석: 가격 ↔ 거래량 선행 가능성(조건부 비교, 실험 A/B)", expanded=True):
    st.caption(
        "방법: 상/하위 분위수로 지역을 구분(중간 구간 제외)한 뒤 분포 차이를 비교합니다. "
        "비모수(Mann–Whitney U) 검정의 p-value로 유의성을 보강합니다."
    )

    q = st.slider("상/하위 분위수 기준(q)", 0.60, 0.85, 0.70, 0.05)

    exp_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["price_growth", "trade_count_growth"]).copy()

    # --- 그룹 생성 ---
    p_hi = exp_df["price_growth"].quantile(q)
    p_lo = exp_df["price_growth"].quantile(1 - q)

    A_high = exp_df[exp_df["price_growth"] >= p_hi].assign(group="가격 변동 상위")
    A_low  = exp_df[exp_df["price_growth"] <= p_lo].assign(group="가격 변동 하위")
    A_df = pd.concat([A_high, A_low], ignore_index=True)

    t_hi = exp_df["trade_count_growth"].quantile(q)
    t_lo = exp_df["trade_count_growth"].quantile(1 - q)

    B_high = exp_df[exp_df["trade_count_growth"] >= t_hi].assign(group="거래량 변동 상위")
    B_low  = exp_df[exp_df["trade_count_growth"] <= t_lo].assign(group="거래량 변동 하위")
    B_df = pd.concat([B_high, B_low], ignore_index=True)

    # --- p-values ---
    uA = mannwhitneyu(A_high["trade_count_growth"], A_low["trade_count_growth"], alternative="two-sided")
    uB = mannwhitneyu(B_high["price_growth"], B_low["price_growth"], alternative="two-sided")
    pA, pB = float(uA.pvalue), float(uB.pvalue)

    # Tooltip (리포트 수준)
    hover_cols = [c for c in [
        "구명",
        "median_price_2023", "median_price_2025",
        "price_growth",
        "trade_share_2023", "trade_share_2025", "trade_share_change",
        "trade_count_2023", "trade_count_2025", "trade_count_growth",
    ] if c in exp_df.columns]

    tab1, tab2, tab3 = st.tabs(["Boxplot", "Slope chart", "Stats"])

    with tab1:
        st.subheader("실험 A: 가격 변동 상/하위 → 거래량 증가율")
        figA_box = px.box(
            A_df,
            x="group",
            y="trade_count_growth",
            points="all",
            hover_data=hover_cols,
            title="실험 A (Boxplot)"
        )
        st.plotly_chart(figA_box, use_container_width=True)

        st.subheader("실험 B: 거래량 변동 상/하위 → 가격 상승률")
        figB_box = px.box(
            B_df,
            x="group",
            y="price_growth",
            points="all",
            hover_data=hover_cols,
            title="실험 B (Boxplot)"
        )
        st.plotly_chart(figB_box, use_container_width=True)

    with tab2:
        st.subheader("실험 A: 가격 변동 상·하위 그룹의 평균 거래지수 변화 (2023=1)")
        st.caption(
            "개별 구의 절대 규모 차이를 제거하기 위해 2023=1로 정규화한 뒤, "
            "가격 변동 상·하위 그룹의 평균적인 거래량 반응(거래지수)을 비교"
        )
    
        # 0) 안전장치
        df_base = df[df["trade_count_2023"] > 0].copy()
    
        # 1) 가격 변동 상·하위 그룹 (상위 30%, 하위 30%)
        p_hi = df_base["price_growth"].quantile(0.7)
        p_lo = df_base["price_growth"].quantile(0.3)
    
        df_A = df_base[(df_base["price_growth"] >= p_hi) | (df_base["price_growth"] <= p_lo)].copy()
        df_A["price_group"] = np.where(df_A["price_growth"] >= p_hi, "가격 변동 상위", "가격 변동 하위")
    
        # 2) 거래지수(정규화): 2023=1, 2025 = trade_count_2025 / trade_count_2023
        df_A["trade_index_2025"] = df_A["trade_count_2025"] / df_A["trade_count_2023"]
    
        # 3) 그룹 평균(또는 중앙값) 계산: 평균이 튀면 median으로 바꿔도 됨
        grp = (
            df_A.groupby("price_group")["trade_index_2025"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
    
        # 4) Slope chart 데이터(2023=1 → 2025=그룹 평균)
        slope_df = pd.concat([
            grp[["price_group", "count"]].assign(year="2023", trade_index=1.0),
            grp[["price_group", "count"]].assign(year="2025", trade_index=grp["median"].values),   # 여기 mean → median 가능
        ], ignore_index=True)
    
        # 5) 그래프 (2개 선만)
        fig_mean = px.line(
            slope_df,
            x="year",
            y="trade_index",
            color="price_group",
            markers=True,
            line_group="price_group",
            hover_data={"count": True, "trade_index": ":.2f", "year": True},
            labels={
                "year": "연도",
                "trade_index": "평균 거래지수 (2023=1)",
                "price_group": "가격 변동 그룹",
                "count": "구 개수",
            },
            title="가격 변동 상·하위 그룹의 평균 거래지수 변화 (2023=1 → 2025)"
        )
    
        fig_mean.update_layout(xaxis=dict(type="category"))
        fig_mean.add_hline(y=1.0, line_dash="dot", opacity=0.6)
    
        # 6) 2025 값 라벨(선 끝에 숫자 표시)
        end_2025 = slope_df[slope_df["year"] == "2025"].copy()
        fig_mean.add_scatter(
            x=end_2025["year"],
            y=end_2025["trade_index"],
            mode="text",
            text=[f"{v:.2f}" for v in end_2025["trade_index"]],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    
        st.plotly_chart(fig_mean, use_container_width=True)
    
        # 7) 숫자 요약(리포트용)
        st.markdown("#### 그룹별 요약 (2025 거래지수)")
        st.dataframe(grp, use_container_width=True)
    
        st.markdown(
            """
            **해석**
            - 2023 기준에서 2025 평균 거래지수가 더 큰 그룹이, 평균적으로 거래량 반응(증가율)이 더 큼  
            - 가격 변동 상위 그룹의 2025 거래지수가 더 크기에 '가격 선행 - 거래량 후행' 가설을 지지  
            - 두 그룹이 유사하면 → 되먹임(상호강화) 또는 외생 요인 가능성 고려
            """
        )

    with tab3:
        c1, c2 = st.columns(2)
        c1.metric("p-value (실험 A)", f"{pA:.3f}")
        c2.metric("p-value (실험 B)", f"{pB:.3f}")

        # 자동 해석 카드
        if pA < 0.05 and pB >= 0.05:
            st.success("해석: 실험 A는 유의, 실험 B는 비유의 → 가격 선행 가능성 시사")
        elif pA < 0.05 and pB < 0.05:
            st.info("해석: 두 실험 모두 유의 → 상호강화(되먹임) 가능성")
        elif pA >= 0.05 and pB < 0.05:
            st.warning("해석: 실험 B만 유의 → 수요(거래량) 선행 가능성")
        else:
            st.error("해석: 두 실험 모두 비유의 → 외생 요인/표본/기간 영향 가능성")

        st.caption(
            "※ 비모수(Mann–Whitney U) 검정 기반. 인과를 단정하지 않고 선행 가능성을 시사하는 수준으로 해석."
        )
