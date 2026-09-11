import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
# 열 이름 찾기 및 통일 함수
# ------------------------------------------------------------
def clean_column_names(df):
    """BOM, 공백 등을 제거하여 열 이름을 정리한다."""
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )
    return df


def find_column(df, candidates):
    """
    후보 열 이름 목록에서 실제 데이터프레임에 있는 열을 찾는다.
    대소문자 차이도 어느 정도 허용한다.
    """
    for col in candidates:
        if col in df.columns:
            return col

    lower_map = {str(col).lower(): col for col in df.columns}

    for col in candidates:
        if str(col).lower() in lower_map:
            return lower_map[str(col).lower()]

    return None


def rename_movie_code_column(df):
    """
    영화코드 열 이름이 movieCd, 영화코드 등 무엇이든
    내부에서는 movieCd로 통일한다.
    """
    movie_code_col = find_column(
        df,
        ["movieCd", "영화코드", "movie_code", "MOVIE_CD"]
    )

    if movie_code_col is None:
        raise ValueError(
            "영화코드 열을 찾을 수 없습니다.\n"
            f"현재 열 목록: {list(df.columns)}"
        )

    if movie_code_col != "movieCd":
        df = df.rename(columns={movie_code_col: "movieCd"})

    df["movieCd"] = df["movieCd"].astype(str).str.strip()

    # CSV를 숫자로 읽었을 때 12345.0처럼 된 경우 처리
    df["movieCd"] = df["movieCd"].str.replace(".0", "", regex=False)

    return df


def find_date_column(df):
    """일별 박스오피스 표에서 날짜 열을 찾는다."""
    return find_column(
        df,
        [
            "날짜",
            "targetDt",
            "date",
            "dt",
            "일자",
            "개봉일"
        ]
    )


def convert_date_to_number(series):
    """
    YYYYMMDD 형태의 날짜를 회귀 모델에서 사용할 수 있는 숫자로 변환한다.
    1970-01-01부터 지난 일수로 바꾼다.
    """
    parsed = pd.to_datetime(
        series.astype(str).str.replace(".0", "", regex=False),
        format="%Y%m%d",
        errors="coerce"
    )

    return (parsed - pd.Timestamp("1970-01-01")).dt.days


# ------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv(DAILY_URL, encoding="utf-8")
    movies = pd.read_csv(MOVIES_URL, encoding="utf-8")

    daily = clean_column_names(daily)
    movies = clean_column_names(movies)

    # 두 파일 모두 영화코드 열을 movieCd로 통일
    daily = rename_movie_code_column(daily)
    movies = rename_movie_code_column(movies)

    return daily, movies


def make_train_test_split(df):
    """
    movieCd 순으로 정렬 후,
    10편마다 앞의 3편을 테스트용으로 사용한다.

    예:
    1~3번: 테스트
    4~10번: 학습
    11~13번: 테스트
    14~20번: 학습
    """
    sorted_df = df.sort_values("movieCd").reset_index(drop=True).copy()

    test_mask = (sorted_df.index % 10) < 3

    test_df = sorted_df[test_mask].copy()
    train_df = sorted_df[~test_mask].copy()

    return train_df, test_df


# ------------------------------------------------------------
# 화면 시작
# ------------------------------------------------------------
st.title("🎬 영화 흥행 예측기")
st.write("영화별 정보 표를 이용하여 영화의 **총 관객 수(total_audi)** 를 예측합니다.")

try:
    daily_df, movies_df = load_data()

except Exception as e:
    st.error("CSV 데이터를 불러오거나 열 이름을 처리하는 중 오류가 발생했습니다.")
    st.error(str(e))

    st.write("일별 표에서 읽힌 열 이름:")
    try:
        raw_daily = pd.read_csv(DAILY_URL, encoding="utf-8")
        st.write(list(raw_daily.columns))
    except Exception:
        pass

    st.stop()


# ------------------------------------------------------------
# 기준 기간 표시
# ------------------------------------------------------------
date_col = find_date_column(daily_df)

if date_col is not None:
    daily_dates = pd.to_datetime(
        daily_df[date_col].astype(str).str.replace(".0", "", regex=False),
        format="%Y%m%d",
        errors="coerce"
    ).dropna()

    if len(daily_dates) > 0:
        start_date = daily_dates.min().strftime("%Y-%m-%d")
        end_date = daily_dates.max().strftime("%Y-%m-%d")
        period_text = f"{start_date} ~ {end_date}"
    else:
        period_text = "날짜 값이 올바르게 읽히지 않았습니다."
else:
    period_text = "날짜 열을 찾을 수 없습니다."

st.info(f"📅 박스오피스 데이터 기준 기간: **{period_text}**")


# ------------------------------------------------------------
# 영화별 표 상위 10행 표시
# ------------------------------------------------------------
st.subheader("영화별 정보 표의 맨 위 10행")
st.dataframe(movies_df.head(10), use_container_width=True)


# ------------------------------------------------------------
# 필요한 열 검사
# ------------------------------------------------------------
if "total_audi" not in movies_df.columns:
    st.error(
        "영화별 표에서 total_audi 열을 찾을 수 없습니다.\n\n"
        f"현재 영화별 표의 열 목록: {list(movies_df.columns)}"
    )
    st.stop()


# ------------------------------------------------------------
# 설명 변수 선택
# ------------------------------------------------------------
st.subheader("학습에 사용할 설명 변수 선택")

possible_features = [
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

feature_options = [
    col for col in possible_features
    if col in movies_df.columns
]

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
    "총 관객 수 예측에 사용할 변수를 선택하세요.",
    options=feature_options,
    default=default_features
)

st.caption(
    "genre와 nation은 범주형 자료로 처리합니다. "
    "openDt와 first_date는 날짜를 숫자로 변환하여 사용합니다."
)

if len(selected_features) == 0:
    st.warning("회귀 모델을 만들려면 적어도 한 개의 변수를 선택해야 합니다.")
    st.stop()


# ------------------------------------------------------------
# 모델 데이터 준비
# ------------------------------------------------------------
model_df = movies_df.copy()

# 총 관객 수를 숫자로 변환
model_df["total_audi"] = pd.to_numeric(
    model_df["total_audi"],
    errors="coerce"
)

# 날짜 변수 변환
date_features = [
    col for col in ["openDt", "first_date"]
    if col in selected_features
]

for col in date_features:
    model_df[col] = convert_date_to_number(model_df[col])

# 숫자형으로 사용해야 할 변수 변환
number_features = [
    "first_scrn",
    "first_show",
    "peak",
    "first_week_audi",
    "days_in_top10"
]

for col in number_features:
    if col in selected_features:
        model_df[col] = pd.to_numeric(model_df[col], errors="coerce")

# 목표값이 없는 영화는 정답이 없으므로 제외
missing_target_count = int(model_df["total_audi"].isna().sum())

if missing_target_count > 0:
    st.warning(
        f"총 관객 수(total_audi)가 비어 있는 영화 {missing_target_count}편은 "
        "학습 및 평가에서 제외됩니다."
    )
    model_df = model_df.dropna(subset=["total_audi"]).copy()

if len(model_df) < 10:
    st.error("학습과 평가에 사용할 영화 수가 너무 적습니다.")
    st.stop()


# ------------------------------------------------------------
# 영화코드 정렬 후 분할
# ------------------------------------------------------------
train_df, test_df = make_train_test_split(model_df)

X_train = train_df[selected_features]
y_train = train_df["total_audi"]

X_test = test_df[selected_features]
y_test = test_df["total_audi"]


# ------------------------------------------------------------
# 전처리와 다중 회귀 모델
# ------------------------------------------------------------
categorical_features = [
    col for col in selected_features
    if col in ["genre", "nation"]
]

numeric_features = [
    col for col in selected_features
    if col not in categorical_features
]

transformers = []

if numeric_features:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    transformers.append(
        ("numeric", numeric_transformer, numeric_features)
    )

if categorical_features:
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )
    transformers.append(
        ("categorical", categorical_transformer, categorical_features)
    )

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

# 관객 수는 음수가 될 수 없으므로 0보다 작은 예측은 0으로 바꾼다.
y_pred = np.maximum(y_pred, 0)


# ------------------------------------------------------------
# 평가 점수 계산
# ------------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

y_test_array = y_test.to_numpy()

# 실제 관객 수가 0명인 경우를 제외하고 MAPE 계산
positive_actual_mask = y_test_array > 0

if positive_actual_mask.sum() > 0:
    mape = (
        np.mean(
            np.abs(
                (y_test_array[positive_actual_mask] - y_pred[positive_actual_mask])
                / y_test_array[positive_actual_mask]
            )
        )
        * 100
    )
else:
    mape = np.nan


# ------------------------------------------------------------
# 영화 수와 기간 표시
# ------------------------------------------------------------
st.subheader("학습 · 평가 구성")

count_col1, count_col2, count_col3 = st.columns(3)

count_col1.metric(
    "학습에 사용한 영화 수",
    f"{len(train_df):,}편"
)

count_col2.metric(
    "점수를 평가한 영화 수",
    f"{len(test_df):,}편"
)

count_col3.metric(
    "전체 사용 영화 수",
    f"{len(model_df):,}편"
)

st.caption(
    "분할 기준: movieCd 오름차순 정렬 후, "
    "영화 10편마다 앞의 3편을 테스트용으로 분리했습니다."
)


# ------------------------------------------------------------
# 평가 점수 표시
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


# ------------------------------------------------------------
# 테스트 결과 표
# ------------------------------------------------------------
result_df = test_df[["movieCd", "movieNm", "total_audi"]].copy()

result_df = result_df.rename(
    columns={
        "total_audi": "실제_총관객수"
    }
)

result_df["예측_총관객수"] = np.round(y_pred).astype(int)
result_df["오차(예측-실제)"] = (
    result_df["예측_총관객수"] - result_df["실제_총관객수"]
)
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
# Plotly 로그 산점도
# ------------------------------------------------------------
st.subheader("실제 총 관객 수와 예측 총 관객 수 비교")

PLOT_FLOOR = 1000

actual_values = y_test.to_numpy()
predicted_values = y_pred.copy()

# 로그 축에는 0을 표시할 수 없으므로 실제값은 최소 1로 처리
actual_plot = np.maximum(actual_values, 1)

# 예측값 1,000명 미만 영화는 그래프 바닥(y=1,000)에 표시
low_prediction_mask = predicted_values < PLOT_FLOOR
predicted_plot = np.maximum(predicted_values, PLOT_FLOOR)

low_prediction_count = int(low_prediction_mask.sum())

all_values = np.concatenate([actual_plot, predicted_plot])

diagonal_min = max(1, float(np.min(all_values)))
diagonal_max = float(np.max(all_values))

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=actual_plot,
        y=predicted_plot,
        mode="markers",
        name="테스트 영화",
        text=result_df["movieNm"],
        customdata=np.column_stack([
            result_df["movieCd"].astype(str),
            actual_values,
            predicted_values,
            result_df["오차(예측-실제)"]
        ]),
        marker=dict(
            size=10,
            color=np.where(low_prediction_mask, "#E74C3C", "#1F77B4"),
            opacity=0.8
        ),
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

# 실제값과 예측값이 같은 위치를 연결하는 대각선
fig.add_trace(
    go.Scatter(
        x=[diagonal_min, diagonal_max],
        y=[diagonal_min, diagonal_max],
        mode="lines",
        name="실제값 = 예측값",
        line=dict(color="black", dash="dash")
    )
)

fig.update_layout(
    template="plotly_white",
    height=650,
    xaxis_title="실제 총 관객 수 (로그 축)",
    yaxis_title="예측 총 관객 수 (로그 축)",
    legend_title="범례"
)

fig.update_xaxes(type="log")
fig.update_yaxes(type="log")

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"빨간 점은 예측 관객 수가 1,000명보다 작은 영화입니다. "
    f"로그 축에서는 0 또는 매우 작은 값을 표현하기 어려워, "
    f"그래프의 바닥값인 1,000명 위치에 표시했습니다. "
    f"해당 영화 수: **{low_prediction_count}편**"
)

st.markdown("---")
st.write(
    "학습 변수 선택을 바꾸어 보며 어떤 영화 정보가 흥행 예측에 "
    "더 도움이 되는지 탐구해 보세요."
)
