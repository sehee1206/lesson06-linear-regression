import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="서울 연평균 기온 예측기", page_icon="🌡️", layout="wide"
)

st.title("🌡️ 서울 연평균 기온 분석 및 예측기")
st.markdown(
    "서울 기온 데이터를 바탕으로 연도별 평균기온 추세를 분석하고, 회귀 직선을 통해 미래의 기온을 예측합니다."
)


# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_and_preprocess_data():
  url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
  df = pd.read_csv(url, encoding="utf-8")

  # 날짜 형식 변환 및 연도 추출
  df["날짜"] = pd.to_datetime(df["날짜"])
  df["연도"] = df["날짜"].dt.year

  # 1. 수업 기준 기간(2025년까지) 필터링
  df_filtered = df[df["연도"] <= 2025]

  # 2. 연도별 평균기온 및 관측일수 계산
  annual_df = (
      df_filtered.groupby("연도")
      .agg(평균기온=("평균기온", "mean"), 관측일수=("평균기온", "count"))
      .reset_index()
  )

  # 3. 관측일이 300일 미만인 해 제외
  valid_df = annual_df[annual_df["관측일수"] >= 300].dropna(
      subset=["평균기온"]
  )
  valid_df = valid_df.sort_values("연도").reset_index(drop=True)

  return valid_df


# 데이터 불러오기
try:
  df_clean = load_and_preprocess_data()
except Exception as e:
  st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
  st.stop()

# 통계 정보 계산
start_year = int(df_clean["연도"].min())
end_year = int(df_clean["연도"].max())
num_years = len(df_clean)

x = df_clean["연도"].values
y = df_clean["평균기온"].values

# 선형 회귀 분석 (1차 다항식 피팅)
slope, intercept = np.polyfit(x, y, 1)
reg_y = slope * x + intercept

# 상관계수 계산
correlation = np.corrcoef(x, y)[0, 1]

# 화면 상단 요약 정보 표시
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석에 사용된 해의 개수", f"{num_years}년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("상관계수 (r)", f"{correlation:.3f}")
st.markdown("---")

# Plotly 그래프 생성
fig = go.Figure()

# 산점도 추가
fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        mode="markers",
        name="연평균 기온 (실측)",
        marker=dict(size=8, color="royalblue"),
    )
)

# 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=x,
        y=reg_y,
        mode="lines",
        name=f"추세선 (y = {slope:.4f}x + {intercept:.2f})",
        line=dict(color="red", width=2),
    )
)

fig.update_layout(
    title="서울 연도별 평균기온 추이 및 회귀 직선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

# 기온 예측 슬라이더 섹션
st.subheader("🔮 연도별 예상 기온 조회")
selected_year = st.slider(
    "예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026
)

# 선택된 연도 기온 예측 계산
predicted_temp = slope * selected_year + intercept

# 결과 강조 표시
st.markdown(
    f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center; border: 2px solid #ff4b4b;">
        <h3 style="margin: 0; color: #31333F;">📅 선택한 연도: {selected_year}년</h3>
        <h1 style="margin: 10px 0; color: #ff4b4b; font-size: 48px;">{predicted_temp:.2f} °C</h1>
        <p style="margin: 0; color: #666666;">회귀 추세선을 기반으로 산출된 예상 연평균 기온입니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
