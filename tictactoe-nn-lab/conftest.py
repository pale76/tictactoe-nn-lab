# pytest がリポジトリ直下の src/ を import パスに加えるためのファイル。
# これにより、インストールなしで `python -m pytest` だけで tictactoe_nn を読み込める。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
