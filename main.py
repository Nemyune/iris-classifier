"""
Iris classifier - 機械学習の最小例
アヤメの花の特徴から品種を予測する
"""

import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
"""
Iris classifier - 機械学習の最小例
アヤメの花の特徴から品種を予測する
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


MODEL_PATH = Path("models/iris_model.pkl")


def train():
    """モデルを学習して保存"""
    # データ読み込み
    iris = load_iris()
    X, y = iris.data, iris.target

    # 訓練/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 学習
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 評価
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"精度: {accuracy:.2%}")
    print("\n詳細レポート:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # 可視化用フォルダ作成
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # 可視化: 特徴量の重要度
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(iris.feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=120)
    plt.close()

    # 可視化: 混同行列
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(3), iris.target_names)
    plt.yticks(range(3), iris.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=120)
    plt.close()

    print(f"グラフを保存しました: {output_dir}/")

    # モデル保存
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"モデルを保存しました: {MODEL_PATH}")


def predict(features: list[float]) -> str:
    """保存済みモデルで予測"""
    iris = load_iris()
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    pred = model.predict([features])[0]
    return iris.target_names[pred]


if __name__ == "__main__":
    train()

    # 試し予測
    sample = [5.1, 3.5, 1.4, 0.2]  # setosa（らしき値）
    result = predict(sample)
    print(f"\n試し予測: {sample} → {result}")