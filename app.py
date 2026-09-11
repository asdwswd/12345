import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="영화 흥행 예측기",
    page_icon="🎬",
    layout="wide"
)

DAILY_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_daily.csv"
MOVIES_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"


# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")

    # movieCd를 문자열로 통일하여 정렬 및 연결 시 안전하게 사용
    daily["movieCd"] = daily["movieCd"].astype(str)
    movies["movieCd"] = movies["movieCd"].astype(str)

    return daily, movies


def find_date_column(df):
    """일별 박스오피스 표에서 날짜 열 이름을 찾는다."""
    candidates = ["날짜", "date", "targetDt", "dt"]

    for col in candidates:
        if col in df.columns:
            return col

    # 이름에 date 또는 날짜가 들어 있는 열도 탐색
    for col in df.columns:
        if "date" in col.lower() or "날짜" in col:
            return col

    return None


def convert_date_to_number(series):
    """
    YYYYMMDD 형식의 날짜를 숫자형 날짜로 변환한다.
    날짜가 비어 있으면 결측값(NaN)이 된다.
    """
    parsed = pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")

    # 1970-01-01로부터 지난 일수 형태의 숫자로 변환
    result = (parsed - pd.Timestamp("1970-01-01")).dt.days
    return result


def make_train_test_split(df):
    """
    영화코드 순서대로 정렬한 뒤,
    열 편(10편)마다 앞의 세 편을 테스트 데이터로 사용한다.

    예:
    1~3번째 영화: 테스트
    4~10번째 영화: 학습
    11~13번째 영화: 테스트
    ...
    """
    sorted_df = df.sort_values("movieCd").reset_index(drop=True).copy()

    test_mask = (sorted_df.index % 10) < 3

    test_df = sorted_df[test_mask].copy()
    train_df = sorted_df[~test_mask].copy()

    return train_df, test_df


# ------------------------------------------------------------
# 화면 제목
# ------------------------------------------------------------
st.title("🎬 영화 흥행 예측기")
st.write("영화 정보와 다중 회귀 모델을 이용하여 영화의 **총 관객 수**를 예측합니다.")

try:
    daily_df, movies_df = load_data()
except Exception as e:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# ------------------------------------------------------------
# 기준 기간 표시
# ------------------------------------------------------------
date_col = find_date_column(daily_df)

if date_col is not None:
    daily_dates = pd.to_datetime(
        daily_df[date_col].astype(str),
        format="%Y%m%d",
        errors="coerce"
    ).dropna()

    if len(daily_dates) > 0:
        start_date = daily_dates.min().strftime("%Y-%m-%d")
        end_date = daily_dates.max().strftime("%Y-%m-%d")
        period_text = f"{start_date} ~ {end_date}"
    else:
        period_text = "날짜 정보를 읽을 수 없습니다."
else:
    period_text = "날짜 열을 찾을 수 없습니다."

st.info(f"📅 박스오피스 데이터 기준 기간: **{period_text}**")


# ------------------------------------------------------------
# 영화별 표 상위 10행 표시
# ------------------------------------------------------------
st.subheader("영화별 정보 표: 맨 위 10행")
st.dataframe(movies_df.head(10), use_container_width=True)


# ------------------------------------------------------------
# 설명 변수 선택
# ------------------------------------------------------------
st.subheader("학습에 사용할 변수 선택")

# 영화코드, 영화명, 목표 변수(total_audi)는 제외
feature_options = [
    "openDt",
    "genre",
    "nation",
    "first_scrn",
    "first_show",
    "first_date",
    "peak",
    "first_week_audi",
    "days_in_top10"
]

# 실제 파일에 있는 열만 사용
feature_options = [col for col in feature_options if col in movies_df.columns]

default_features = [
    col for col in [
        "genre",
        "nation",
        "first_scrn",
        "first_show",
        "peak",
        "first_week_audi",
        "days_in_top10"
    ]
    if col in feature_options
]

selected_features = st.multiselect(
    "총 관객 수를 예측할 때 사용할 설명 변수를 선택하세요.",
    options=feature_options,
    default=default_features
)

st.caption(
    "※ openDt와 first_date는 날짜를 숫자형 날짜로 변환하여 사용합니다. "
    "genre와 nation은 범주형 변수로 처리합니다."
)

if len(selected_features) == 0:
    st.warning("학습을 위해 적어도 한 개의 설명 변수를 선택하세요.")
    st.stop()

if "total_audi" not in movies_df.columns:
    st.error("영화별 표에 total_audi(총 관객 수) 열이 없습니다.")
    st.stop()


# ------------------------------------------------------------
# 모델용 데이터 준비
# ------------------------------------------------------------
model_df = movies_df.copy()

# 목표 변수 숫자 변환
model_df["total_audi"] = pd.to_numeric(model_df["total_audi"], errors="coerce")

# 날짜 변수는 회귀 분석에 사용할 수 있도록 숫자 날짜로 변환
date_features = [col for col in ["openDt", "first_date"] if col in selected_features]

for col in date_features:
    model_df[col] = convert_date_to_number(model_df[col])

# 수치형 변수와 범주형 변수 구분
categorical_features = [
    col for col in selected_features
    if col in ["genre", "nation"]
]

numeric_features = [
    col for col in selected_features
    if col not in categorical_features
]

# total_audi가 없는 행은 정답이 없으므로 평가가 불가능하다.
# 데이터에 총 관객 수가 있는 모든 영화는 학습/평가에 사용한다.
missing_target_count = model_df["total_audi"].isna().sum()

if missing_target_count > 0:
    st.warning(
        f"총 관객 수(total_audi)가 없는 영화 {missing_target_count}편은 "
        "정답이 없어 학습과 평가에서 제외됩니다."
    )
    model_df = model_df.dropna(subset=["total_audi"]).copy()

if len(model_df) < 10:
    st.error("학습과 평가에 사용할 영화 수가 충분하지 않습니다.")
    st.stop()


# ------------------------------------------------------------
# 영화코드 순 정렬 후 10편마다 앞 3편을 테스트로 분리
# ------------------------------------------------------------
train_df, test_df = make_train_test_split(model_df)

X_train = train_df[selected_features]
y_train = train_df["total_audi"]

X_test = test_df[selected_features]
y_test = test_df["total_audi"]

# 숫자형 처리: 결측값 중앙값 대체 + 표준화
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# 범주형 처리: 결측값 최빈값 대체 + 원-핫 인코딩
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

transformers = []

if len(numeric_features) > 0:
    transformers.append(("num", numeric_transformer, numeric_features))

if len(categorical_features) > 0:
    transformers.append(("cat", categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(transformers=transformers)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)

# ------------------------------------------------------------
# 학습 및 예측
# ------------------------------------------------------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 관객 수는 음수가 될 수 없으므로 음수 예측값은 0으로 처리
y_pred = np.maximum(y_pred, 0)


# ------------------------------------------------------------
# 평가 점수
# ------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

# 실제값이 0보다 큰 경우에만 백분율 오차 계산
nonzero_mask = y_test > 0
if nonzero_mask.sum() > 0:
    mape = (
        np.mean(
            np.abs(
                (y_test[nonzero_mask].to_numpy() - y_pred[nonzero_mask.to_numpy()])
                / y_test[nonzero_mask].to_numpy()
            )
        )
        * 100
    )
else:
    mape = np.nan


# ------------------------------------------------------------
# 학습 및 평가 영화 수 표시
# ------------------------------------------------------------
st.subheader("학습 · 평가 구성")

col1, col2, col3 = st.columns(3)
col1.metric("학습에 사용한 영화 수", f"{len(train_df):,}편")
col2.metric("점수를 평가한 테스트 영화 수", f"{len(test_df):,}편")
col3.metric("전체 사용 영화 수", f"{len(model_df):,}편")

st.caption(
    "분할 방법: movieCd(영화코드) 오름차순 정렬 후, "
    "10편마다 앞의 3편은 테스트용 · 나머지 7편은 학습용"
)


# ------------------------------------------------------------
# 평가 지표 표시
# ------------------------------------------------------------
st.subheader("테스트 영화 평가 점수")

score_col1, score_col2, score_col3, score_col4 = st.columns(4)

score_col1.metric("결정계수 R²", f"{r2:.3f}")
score_col2.metric("평균 절대 오차 (MAE)", f"{mae:,.0f}명")
score_col3.metric("평균 제곱근 오차 (RMSE)", f"{rmse:,.0f}명")

if np.isnan(mape):
    score_col4.metric("평균 절대 백분율 오차 (MAPE)", "계산 불가")
else:
    score_col4.metric("평균 절대 백분율 오차 (MAPE)", f"{mape:.1f}%")

st.write(
    "- **MAE**: 예측값이 실제 관객 수와 평균적으로 몇 명 차이 나는지 나타냅니다.\n"
    "- **RMSE**: 큰 오차에 더 큰 벌점을 주는 오차 지표입니다.\n"
    "- **R²**: 1에 가까울수록 실제 관객 수의 변화를 더 잘 설명합니다."
)


# ------------------------------------------------------------
# 테스트 영화별 예측 결과와 오차 표
# ------------------------------------------------------------
result_df = test_df[["movieCd", "movieNm", "total_audi"]].copy()
result_df = result_df.rename(columns={"total_audi": "실제_총관객수"})

result_df["예측_총관객수"] = np.round(y_pred).astype(int)
result_df["오차(예측-실제)"] = result_df["예측_총관객수"] - result_df["실제_총관객수"]
result_df["절대오차"] = np.abs(result_df["오차(예측-실제)"])

result_df = result_df.sort_values("movieCd").reset_index(drop=True)

st.subheader("테스트 영화별 실제값 · 예측값 · 오차")
st.dataframe(
    result_df.style.format({
        "실제_총관객수": "{:,.0f}",
        "예측_총관객수": "{:,.0f}",
        "오차(예측-실제)": "{:+,.0f}",
        "절대오차": "{:,.0f}"
    }),
    use_container_width=True
)


# ------------------------------------------------------------
# Plotly 산점도
# ------------------------------------------------------------
st.subheader("실제 총 관객 수와 예측 총 관객 수 비교")

# 로그 축에서는 0 이하 값을 표현할 수 없으므로,
# 예측값이 1,000명 미만인 영화는 그래프 바닥값 1,000에 붙여 표시한다.
PLOT_FLOOR = 1000

actual_values = y_test.to_numpy()
predicted_values = y_pred.copy()

# 실제값도 로그 축을 위해 최소 1명 처리
actual_plot = np.maximum(actual_values, 1)

low_prediction_mask = predicted_values < PLOT_FLOOR
predicted_plot = np.maximum(predicted_values, PLOT_FLOOR)

low_prediction_count = int(low_prediction_mask.sum())

# 대각선 범위 설정
all_plot_values = np.concatenate([actual_plot, predicted_plot])
diag_min = max(1, min(all_plot_values))
diag_max = max(all_plot_values)

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=actual_plot,
        y=predicted_plot,
        mode="markers",
        name="테스트 영화",
        marker=dict(
            size=10,
            color=np.where(low_prediction_mask, "#e74c3c", "#1f77b4"),
            opacity=0.8
        ),
        text=result_df["movieNm"],
        customdata=np.column_stack([
            result_df["movieCd"],
            actual_values,
            predicted_values,
            result_df["오차(예측-실제)"]
        ]),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "영화코드: %{customdata[0]}<br>"
            "실제 총 관객 수: %{customdata[1]:,.0f}명<br>"
            "예측 총 관객 수: %{customdata[2]:,.0f}명<br>"
            "오차: %{customdata[3]:+,.0f}명"
            "<extra></extra>"
        )
    )
)

# 실제값 = 예측값인 대각선
fig.add_trace(
    go.Scatter(
        x=[diag_min, diag_max],
        y=[diag_min, diag_max],
        mode="lines",
        name="실제값 = 예측값",
        line=dict(color="black", dash="dash")
    )
)

fig.update_layout(
    height=650,
    xaxis_title="실제 총 관객 수 (로그 축)",
    yaxis_title="예측 총 관객 수 (로그 축)",
    legend_title="범례",
    template="plotly_white"
)

fig.update_xaxes(type="log")
fig.update_yaxes(type="log")

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"빨간 점은 예측 총 관객 수가 1,000명보다 작은 영화입니다. "
    f"로그 축에 표시하기 위해 그래프에서는 y=1,000 위치에 붙였습니다. "
    f"해당 영화 수: **{low_prediction_count}편**"
)

st.markdown("---")
st.write(
    "💡 변수 선택을 바꾸어 보면서 어떤 영화 정보가 총 관객 수 예측에 "
    "도움이 되는지 비교해 보세요."
)
