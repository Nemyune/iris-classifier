# Python 3.12 slim版をベースに
FROM python:3.12-slim

# 作業ディレクトリ
WORKDIR /app

# uvをインストール
RUN pip install --no-cache-dir uv

# 依存関係ファイルだけ先にコピー（キャッシュ効率化）
COPY pyproject.toml uv.lock ./

# 依存関係インストール
RUN uv sync --frozen --no-dev

# プロジェクト全体をコピー
COPY . .

# 実行
CMD ["uv", "run", "python", "main.py"]
