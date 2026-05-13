"""Streamlit アヤメ分類デモ"""

import pickle
from pathlib import Path

import streamlit as st
from sklearn.datasets import load_iris


MODEL_PATH = Path("models/iris_model.pkl")


@st.cache_resource
def load_model():
    """モデル読み込み（キャッシュ）"""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def main():
    st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
    st.title("🌸 アヤメ品種分類デモ")
    st.write("4つの特徴量を入力すると、アヤメの品種を予測します。")

    iris = load_iris()
    model = load_model()

    # サイドバーで入力
    st.sidebar.header("特徴量を入力")
    sepal_length = st.sidebar.slider("がく片の長さ (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.sidebar.slider("がく片の幅 (cm)", 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider("花弁の長さ (cm)", 1.0, 7.0, 1.4)
    petal_width = st.sidebar.slider("花弁の幅 (cm)", 0.1, 2.5, 0.2)

    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    # 予測
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    species = iris.target_names[pred]

    # 結果表示
    st.subheader("予測結果")
    st.success(f"**品種: {species}**")

    # 確率を棒グラフで表示
    st.subheader("各品種の確率")
    proba_dict = {name: p for name, p in zip(iris.target_names, proba)}
    st.bar_chart(proba_dict)

    # 入力値の確認
    with st.expander("入力値の確認"):
        st.json({
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        })


if __name__ == "__main__":
    main()
