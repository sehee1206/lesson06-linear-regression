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
    "서울 기온 데이터를 바탕으로 연도별 평균기온 추세를 분석하고, 전체 기간과 최근 20년의 기온 상승 속도를 비교합니다."
)


# 데이터 로드 및 전처리 함수 (캐싱 적용)
@st.cache_data
def load_and_preprocess_data():
  url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
  df = pd.read_csv(url, encoding="utf-8")

  # 날짜 형식 변환 및 연도 추출
  df["날짜"] = pd.to_datetime(df["날짜"])
  df["연도"] = df["날짜"].dt.year

  # 수업 기준 기간(2025년까지) 필터링
  df_filtered = df[df["연도"] <= 2025]

  # 연도별 평균기온 및 관측일수 계산
  annual_df = (
      df_filtered.groupby("연도")
      .agg(평균기온=("평균기온", "mean"), 관측일수=("평균기온", "count"))
      .reset_index()
  )

  # 관측일이 300일 미만인 해 제외
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

# 전체 기간 통계 정보
start_year = int(df_clean["연도"].min())
end_year = int(df_clean["연도"].max())
num_years = len(df_clean)

x_all = df_clean["연도"].values
y_all = df_clean["평균기온"].values

# 전체 기간 회귀 분석 및 상관계수
slope_all, intercept_all = np.polyfit(x_all, y_all, 1)
reg_y_all = slope_all * x_all + intercept_all
correlation_all = np.corrcoef(x_all, y_all)[0, 1]

# 최근 20년 기간 데이터 필터링 (예: 2006년 ~ 2025년 등 데이터상 최근 20개년)
df_recent = df_clean[df_clean["연도"] >= (end_year - 19)]
x_recent = df_recent["연도"].values
y_recent = df_recent["평균기온"].values
slope_recent, intercept_recent = np.polyfit(x_recent, y_recent, 1)
reg_y_recent = slope_recent * x_recent + intercept_recent
correlation_recent = np.corrcoef(x_recent, y_recent)[0, 1]

# 100년 당 상승 온도 환산 (기울기 * 100)
temp_change_100y_all = slope_all * 100
temp_change_100y_recent = slope_recent * 100

# 화면 상단 요약 정보 표시
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석에 사용된 해의 개수", f"{num_years}년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("전체 기간 상관계수 (r)", f"{correlation_all:.3f}")
st.markdown("---")

# 📌 기울기 비교 섹션 (100년 당 온도 상승폭)
st.subheader("📈 기온 상승 속도 비교 (100년 기준 환산)")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
  st.markdown(
      f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #f8f9fa; text-align: center; border: 1px solid #dcdcdc;">
            <h4 style="margin: 0; color: #31333F;">🌍 전체 기간 ({start_year}~{end_year}년)</h4>
            <h2 style="margin: 10px 0; color: #1f77b4; font-size: 36px;">+{temp_change_100y_all:.2f} °C / 100년</h2>
            <p style="margin: 0; color: #666666; font-size: 14px;">연간 기울기: {slope_all:.4f} °C/년</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

with comp_col2:
  st.markdown(
      f"""
        <div style="padding: 20px; border-radius: 10px; background-color: #fff0f0; text-align: center; border: 1px solid #ff4b4b;">
            <h4 style="margin: 0; color: #31333F;">🔥 최근 20년 ({int(x_recent[0])}~{int(x_recent[-1])}년)</h4>
            <h2 style="margin: 10px 0; color: #ff4b4b; font-size: 36px;">+{temp_change_100y_recent:.2f} °C / 100년</h2>
            <p style="margin: 0; color: #666666; font-size: 14px;">연간 기울기: {slope_recent:.4f} °C/년</p>
        </div>
        """,
      unsafe_allow_html=True,
  )

st.markdown("<br>", unsafe_allow_html=True)

# Plotly 그래프 생성
fig = go.Figure()

# 전체 산점도 추가
fig.add_trace(
    go.Scatter(
        x=x_all,
        y=y_all,
        mode="markers",
        name="연평균 기온 (실측)",
        marker=dict(size=7, color="lightblue"),
    )
)

# 전체 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=x_all,
        y=reg_y_all,
        mode="lines",
        name=f"전체 추세선 (100년당 +{temp_change_100y_all:.2f}°C)",
        line=dict(color="royalblue", width=2),
    )
)

# 최근 20년 회귀 직선 추가
fig.add_trace(
    go.Scatter(
        x=x_recent,
        y=reg_y_recent,
        mode="lines",
        name=f"최근 20년 추세선 (100년당 +{temp_change_100y_recent:.2f}°C)",
        line=dict(color="red", width=2, dash="dash"),
    )
)

fig.update_layout(
    title="서울 연도별 평균기온 추이 및 회귀 직선 비교",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    template="plotly_white",
    height=550,
)

st.plotly_chart(fig, use_container_width=True)

# 기온 예측 슬라이더 섹션
st.subheader("🔮 연도별 예상 기온 조회 (전체 회귀선 기준)")
selected_year = st.slider(
    "예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026
)

# 선택된 연도 기온 예측 계산 (전체 추세선 기준)
predicted_temp = slope_all * selected_year + intercept_all

# 결과 강조 표시
st.markdown(
    f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center; border: 2px solid #1f77b4;">
        <h3 style="margin: 0; color: #31333F;">📅 선택한 연도: {selected_year}년</h3>
        <h1 style="margin: 10px 0; color: #1f77b4; font-size: 48px;">{predicted_temp:.2f} °C</h1>
        <p style="margin: 0; color: #666666;">전체 기간 회귀 추세선을 기반으로 산출된 예상 연평균 기온입니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
