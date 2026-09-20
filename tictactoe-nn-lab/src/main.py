"""アプリの起動用スクリプト。

使い方:
    python src/main.py                # ローカルで起動（表示されたURLをブラウザで開く）
    python src/main.py --port 8000    # ポートを指定
    python src/main.py --share        # 公開リンクを作る（リンクを知っている人は誰でも開ける）
"""
import argparse
import sys
from pathlib import Path

# どのフォルダから実行しても、同じ src/ 内の tictactoe_nn を読み込めるようにする
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tictactoe_nn.app import launch  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="〇×ゲーム ニューラルネットワーク実験ラボ")
    parser.add_argument("--share", action="store_true",
                        help="公開リンク（*.gradio.live）を作る。指定しなければ Gradio の既定に従う")
    parser.add_argument("--port", type=int, default=None, help="使用するポート番号")
    args = parser.parse_args(argv)

    kwargs = {}
    if args.port is not None:
        kwargs["server_port"] = args.port
    launch(share=True if args.share else None, **kwargs)


if __name__ == "__main__":
    main()
