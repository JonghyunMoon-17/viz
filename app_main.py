import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import linregress, mannwhitneyu

st.set_page_config(layout="wide")

# -----------------------------
# 0) 유틸: CSV 로드 (인코딩 안전)
# -----------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # utf-8-sig 우선, 실패하면 기본
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path)
    return df

# -----------------------------
# 1) 급지(1~4) 구 리스트 (※ 너 기준으로 나중에 수정하면 됨)
# -----------------------------
GRADE_MAP = {
    "전체": [],
    # ⚠️ 아래는 임시안. 너희 팀이 정의한 1~4급지 기준으로 바꾸면 됨.
    "1급지": ["강남구", "서초구", "송파구", "용산구", "성동구"],
    "2급지": ["마포구", "동작구", "영등포구", "광진구", "강동구", "서대문구"],
    "3급지": ["종로구", "중구", "동대문구", "강서구", "양천구", "성북구"],
    "4급지": ["노원구", "도봉구", "강북구", "금천구", "구로구", "중랑구", "은평구"],
}

# -----------------------------
# 2) 데이터 로드 & 기본 정리
# -----------------------------
df = load_data("district_summary.csv").copy()

# 필수 컬럼 체크(없으면 생성)
required_cols = [
    "구명",
    "median_price_2023", "median_price_2025",
    "price_growth",
    "trade_share_2023", "trade_share_2025", "trade_share_change",
    "trade_count_2023", "trade_count_2025"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"district_summary.csv에 필수 컬럼이 없습니다: {missing}")
    st.stop()

# 거래건수 증가율 없으면 생성
if "trade_count_growth" not in df.columns:
    df["trade_count_growth"] = (df["trade_count_2025"] - df["trade_count_2023"]) / df["trade_count_2023"].replace(0, np.nan)

# 안전 필터
df = df[df["trade_count_2023"] > 0].copy()
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=["price_growth", "trade_count_2025"]).copy()

# -----------------------------
# 3) 상단 UI: 지표 모드 + 급지 강조 선택
# -----------------------------
st.title("가격 상승률 vs 거래 변화 (통합 시각화)")
st.caption("절대지표(거래건수)와 상대지표(거래비중)를 버튼 하나로 전환하며, 2025 거래건수(버블 크기) 기반으로 분포를 함께 확인합니다.")

left_ui, right_ui = st.columns([1.2, 1])

with left_ui:
    mode = st.radio(
        "지표 모드 선택",
        ["절대지표(거래건수)", "상대지표(거래비중)"],
        horizontal=True
    )

with right_ui:
    grade_key = st.selectbox("급지 강조(1~4급지)", list(GRADE_MAP.keys()), index=0)

selected_gus = set(GRADE_MAP.get(grade_key, []))

# 강조 마스크
if grade_key == "전체" or len(selected_gus) == 0:
    df["강조"] = "전체"
else:
    df["강조"] = np.where(df["구명"].isin(selected_gus), "선택 급지", "기타")

# -----------------------------
# 4) 메인 산점도: 절대/상대 토글
# -----------------------------
if mode == "절대지표(거래건수)":
    y_col = "trade_count_growth"
    y_label = "거래건수 증가율 (2023→2025)"
else:
    y_col = "trade_share_change"
    y_label = "거래 비중 변화 (2023→2025)"

# y 결측 제거
plot_df = df.dropna(subset=["price_growth", y_col]).copy()

# -----------------------------
# 5) 회귀/통계 요약 (r, beta, R2, p-value)
# -----------------------------
x = plot_df["price_growth"].astype(float).values
y = plot_df[y_col].astype(float).values

# 선형회귀 (y = a + b x)
lr = linregress(x, y)
r = lr.rvalue
beta = lr.slope
r2 = lr.rvalue ** 2
p_beta = lr.pvalue

# -----------------------------
# 6) Hover 템플릿 (한국어 + 순서 고정 + label 제거)
#   Plotly의 hovertemplate은 customdata로 순서 제어 가능
# -----------------------------
# 공통 커스텀 데이터 구성 (순서를 우리가 고정)
# [0] 2023 중위가격
# [1] 2025 중위가격
# [2] 가격상승률
# [3] 2023 거래비중
# [4] 2025 거래비중
# [5] 거래비중 변화
# [6] 2023 거래건수
# [7] 2025 거래건수
# [8] 거래건수 증가율
customdata = np.stack([
    plot_df["median_price_2023"].values,
    plot_df["median_price_2025"].values,
    plot_df["price_growth"].values,
    plot_df["trade_share_2023"].values,
    plot_df["trade_share_2025"].values,
    plot_df["trade_share_change"].values,
    plot_df["trade_count_2023"].values,
    plot_df["trade_count_2025"].values,
    plot_df["trade_count_growth"].values,
], axis=1)

# 모드별 hovertemplate
if mode == "절대지표(거래건수)":
    hovertemplate = (
        "<b>%{hovertext}</b><br><br>"
        "2023 중위가격: %{customdata[0]:,.0f}<br>"
        "2025 중위가격: %{customdata[1]:,.0f}<br>"
        "가격 상승률(2023→2025): %{customdata[2]:.3f}<br><br>"
        "2023 거래건수: %{customdata[6]:,.0f}<br>"
        "2025 거래건수: %{customdata[7]:,.0f}<br>"
        "거래건수 증가율(2023→2025): %{customdata[8]:.2%}<br>"
        "<extra></extra>"
    )
else:
    hovertemplate = (
        "<b>%{hovertext}</b><br><br>"
        "2023 중위가격: %{customdata[0]:,.0f}<br>"
        "2025 중위가격: %{customdata[1]:,.0f}<br>"
        "가격 상승률(2023→2025): %{customdata[2]:.3f}<br><br>"
        "2023 거래비중: %{customdata[3]:.3f}<br>"
        "2025 거래비중: %{customdata[4]:.3f}<br>"
        "거래 비중 변화(2023→2025): %{customdata[5]:.3f}<br><br>"
        "2023 거래건수: %{customdata[6]:,.0f}<br>"
        "2025 거래건수: %{customdata[7]:,.0f}<br>"
        "<extra></extra>"
    )

# -----------------------------
# 7) 메인 차트 생성 (버블 size = 2025 거래건수)
#   강조 선택 시: 선택 급지만 진하게, 기타는 연하게
# -----------------------------
fig = px.scatter(
    plot_df,
    x="price_growth",
    y=y_col,
    size="trade_count_2025",
    size_max=55,
    hover_name="구명",
    color="강조",
    color_discrete_map={
        "전체": "#1f77b4",      # 기본 파란 계열(Plotly 기본과 조화)
        "선택 급지": "#1f77b4", # 같은 색, 대신 opacity로 강조
        "기타": "#1f77b4",
    },
    labels={
        "price_growth": "가격 상승률 (2023→2025)",
        y_col: y_label,
    },
)

# customdata + hovertemplate 적용 (label=... 같은 불필요정보 제거)
fig.update_traces(
    customdata=customdata,
    hovertemplate=hovertemplate,
)

# 강조 스타일(투명도)
for tr in fig.data:
    if tr.name == "기타":
        tr.update(marker=dict(opacity=0.18))
    else:
        tr.update(marker=dict(opacity=0.90))

# 회귀선은 plotly express trendline 대신, 통계값은 우측에만 보여주고
# 시각적으로는 최소한의 회귀선만 추가(원하면 제거 가능)
# 간단하게 y = intercept + beta*x 선을 추가
x_line = np.linspace(plot_df["price_growth"].min(), plot_df["price_growth"].max(), 50)
y_line = lr.intercept + beta * x_line
fig.add_scatter(
    x=x_line,
    y=y_line,
    mode="lines",
    name="회귀선",
    hoverinfo="skip"
)

# -----------------------------
# 8) 레이아웃: 좌(차트) + 우(요약)
# -----------------------------
col_chart, col_summary = st.columns([2.15, 1])

with col_chart:
    st.subheader("산점도")
    st.plotly_chart(fig, use_container_width=True)

with col_summary:
    st.subheader("해석 요약(통계)")
    st.caption("※ 선택한 지표 모드(절대/상대)에 따라 Y축 및 회귀 결과가 자동 업데이트됩니다.")

    # 보기 좋게 포맷
    st.markdown(
        f"""
        - **상관계수 r:** `{r:.3f}`
        - **회귀 기울기 β:** `{beta:.4f}`
        - **R²:** `{r2:.3f}`
        - **p-value (β):** `{p_beta:.4f}`
        """
    )

    if grade_key != "전체" and len(selected_gus) > 0:
        st.markdown("**급지 강조 목록(현재 선택):**")
        st.write(", ".join(GRADE_MAP[grade_key]))

# -----------------------------
# 9) 추가분석(스크롤 아래) — 조건부 비교 A/B (slope 제거, stats만)
#     - 좌우 배치
#     - 결과 약한 쪽 먼저(분석 1), 결과 좋은 쪽 나중(분석 2) + expander로 숨김
# -----------------------------
st.divider()
st.header("추가분석: 가격 ↔ 거래량 선행 가능성(조건부 비교, 분석 1/2)")
st.caption(
    "두 변수를 '상·하위 집단'으로 나눠서 비교합니다. (시각화는 제외하고 통계 요약만 제공합니다.)"
)

# 분석용 데이터
adf = df.dropna(subset=["price_growth", "trade_count_growth"]).copy()

# --- 분석 1 (먼저 보여주기): 거래량 변동(상/하위) 조건 하 가격 상승률 비교 ---
q_hi = adf["trade_count_growth"].quantile(0.7)
q_lo = adf["trade_count_growth"].quantile(0.3)

B_high = adf[adf["trade_count_growth"] >= q_hi].copy()
B_low  = adf[adf["trade_count_growth"] <= q_lo].copy()

uB = mannwhitneyu(B_high["price_growth"], B_low["price_growth"], alternative="two-sided")

def summarize_group(name: str, series: pd.Series) -> dict:
    s = series.dropna()
    return {
        "집단": name,
        "표본수(n)": int(s.shape[0]),
        "평균": float(s.mean()) if len(s) else np.nan,
        "중앙값": float(s.median()) if len(s) else np.nan,
    }

B_sum = pd.DataFrame([
    summarize_group("거래량 변동 상위(Top30%)", B_high["price_growth"]),
    summarize_group("거래량 변동 하위(Bottom30%)", B_low["price_growth"]),
])

# --- 분석 2 (나중 + 펼쳐보기): 가격 변동(상/하위) 조건 하 거래량 증가율 비교 ---
p_hi = adf["price_growth"].quantile(0.7)
p_lo = adf["price_growth"].quantile(0.3)

A_high = adf[adf["price_growth"] >= p_hi].copy()
A_low  = adf[adf["price_growth"] <= p_lo].copy()

uA = mannwhitneyu(A_high["trade_count_growth"], A_low["trade_count_growth"], alternative="two-sided")

A_sum = pd.DataFrame([
    summarize_group("가격 변동 상위(Top30%)", A_high["trade_count_growth"]),
    summarize_group("가격 변동 하위(Bottom30%)", A_low["trade_count_growth"]),
])

# 좌/우 배치
c1, c2 = st.columns(2)

with c1:
    st.subheader("분석 1: 거래량 변동(상/하위) → 가격 상승률")
    st.caption("거래량 증가율 기준으로 상·하위 집단을 만든 뒤, 가격 상승률 분포가 달라지는지 비교")

    st.dataframe(B_sum, use_container_width=True)
    st.markdown(f"- **Mann–Whitney U p-value:** `{uB.pvalue:.4f}`")

    st.markdown(
        """
        **해석 힌트**
        - p-value가 작을수록(예: 0.05 미만) 두 집단의 분포 차이가 뚜렷하다고 볼 수 있음
        - 여기서는 “거래량 변화가 가격을 선행”한다는 주장에 힘을 주는 근거로 해석 가능
        """
    )

with c2:
    st.subheader("분석 2: 가격 변동(상/하위) → 거래량 증가율")
    st.caption("가격 상승률 기준으로 상·하위 집단을 만든 뒤, 거래량 증가율 분포가 달라지는지 비교")

    with st.expander("📌 클릭해서 분석 2 결과 펼치기", expanded=False):
        st.dataframe(A_sum, use_container_width=True)
        st.markdown(f"- **Mann–Whitney U p-value:** `{uA.pvalue:.4f}`")

        st.markdown(
            """
            **해석 힌트**
            - p-value가 충분히 작다면: “가격 변화가 먼저 발생한 지역에서 거래량이 후행 반응했을 가능성”을 시사
            - 단, 인과를 ‘증명’하는 게 아니라 ‘조건부 차이(패턴)’를 보여주는 보강 증거임
            """
        )

# (요청) slope chart, Q1/Q4 박스플롯, lead-lag(Δcorr) 섹션은 app_main.py에 포함하지 않음.
