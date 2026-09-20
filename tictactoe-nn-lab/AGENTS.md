# AGENTS.md

このファイルは、このリポジトリで作業する coding agent（AIコーディングエージェント）と、人間の開発者向けの作業ガイドです。

## AI の利用について

このプロジェクトのコードとドキュメントの初版は、AI アシスタント（Anthropic の Claude）と対話しながら作成しました。
仕様の決定、Google Colab 上での動作確認、実験の実行、公開は、リポジトリの所有者（人間）が行っています。
`docs/experiments.md` の数値は、所有者が実際に Colab で実行した画面の値です。

## プロジェクト概要

〇×ゲームの盤面（9マス）から勝敗を当てるニューラルネットワーク（PyTorch）を学習させ、学習データ数・エポック数・ネットワーク構成による正答率の変化を調べる実験アプリ。Gradio の Web 画面を持ち、Google Colab で動く。

- 詳しい仕様: [docs/design.md](docs/design.md)
- 実験結果: [docs/experiments.md](docs/experiments.md)

## セットアップと主なコマンド

```bash
pip install -r requirements.txt          # 実行に必要なもの
pip install -r requirements-dev.txt      # テスト用（pytest を追加）

python src/main.py                       # アプリを起動
python -m pytest -q                      # テスト（PyTorch が無い環境では、学習まわりは自動でスキップ）
python src/make_sample_data.py           # data/sample.csv を作り直す
```

## ディレクトリ構成

```
src/main.py                 起動用スクリプト
src/tictactoe_nn/game.py    盤面の生成・勝敗ラベル（PyTorch 不要）
src/tictactoe_nn/data.py    学習/テスト分割・入力表現（PyTorch 不要）
src/tictactoe_nn/model.py   NN の定義
src/tictactoe_nn/train.py   学習・評価・複数回実行・学習曲線
src/tictactoe_nn/plots.py   グラフ（matplotlib）
src/tictactoe_nn/app.py     Gradio 画面
src/make_sample_data.py     data/sample.csv の生成
tests/                      自動テスト
data/sample.csv             盤面とラベルのサンプル（全5,478通り）
assets/                     ロゴ・ファビコン
docs/                       設計・実験結果・スクリーンショット
notebooks/colab_demo.ipynb  Colab 用ノートブック
```

## 守ること

**実験の公平さ**
- テストデータを学習に使わない。分割は `data.build_dataset` の固定シード（42）で決まる。これを変えると、過去の実験結果と比べられなくなる。変えるときは、`data/sample.csv` の再生成、テストの更新、`docs/experiments.md` への注記が必要。
- 実験結果の数値を README や docs に書くときは、実際に実行して得た値だけを使う。推測で数値を作らない。仮説は「仮説」と明記する。

**ルール（game.py）**
- 勝敗判定や盤面生成を変えたら、`tests/test_game.py` の期待値（盤面 5,478 通り、P1勝ち 626・P2勝ち 316・引き分け 16）を必ず確認し、`python src/make_sample_data.py` で `data/sample.csv` を作り直す。

**コード**
- `game.py` と `data.py` は PyTorch に依存させない（PyTorch なしでルールのテストを動かすため）。
- Gradio は 5 系・6 系の両方で動くように、`gr.Blocks()` に `theme` や `css` を渡さない。ハンドラーの引数は、`*args` にせず1つずつ明示する。
- グラフは `pyplot` を使わず、`matplotlib.figure.Figure` を直接作る（Gradio とノートブックの両方で使えるため）。日本語フォントが無いときは、英語ラベルに切り替わる仕組みを保つ。
- コメントと docstring、ドキュメントは日本語。識別子は英語。
- 新しい依存ライブラリは `requirements.txt` に追加する。

## 変更後のチェックリスト

- [ ] `python -m pytest -q` が通る
- [ ] `data/sample.csv` が最新（ルールやデータ分割を変えた場合）
- [ ] README・docs の説明が、コードの実際の動作と合っている
- [ ] `notebooks/colab_demo.ipynb` の読み込みセルが、フォルダ構成の変更に追従している

## 既知の制約

- Google Colab では、Gradio が自動で公開リンク（`*.gradio.live`）を作ることがある。リンクを知っている人は誰でも開ける。
- 「全パターン」モードでは、両者が同時に3つ並べている盤面のラベルを便宜上 0 としている。
- 詳しくは [docs/design.md](docs/design.md) の「既知の制約」を参照。
