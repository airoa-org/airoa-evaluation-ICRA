"""VLA-MoE Router: タスク指示テキストから適切な Expert を選択する。

Round 4 用のルールベース Router と、将来の研究用 Learned Router のインターフェースを提供。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseRouter(ABC):
    """Router の基底クラス。"""

    @abstractmethod
    def route(self, instruction: str) -> int:
        """タスク指示テキストから Expert インデックスを返す。

        Args:
            instruction: タスク指示テキスト（例: "Pick the coffee from the table"）

        Returns:
            Expert インデックス（0-indexed）
        """
        ...


class RuleBasedRouter(BaseRouter):
    """キーワードマッチによるルールベース Router。

    公開6タスクのキーワードパターンに基づいて Expert を選択する。
    マッチしない場合はデフォルト Expert（汎用）にフォールバック。
    """

    # タスクキーワード定義: Expert ID → 必須キーワードリスト
    # 全キーワードが instruction に含まれていればマッチ
    DEFAULT_TASK_KEYWORDS: dict[int, list[str]] = {
        0: ["pick", "coffee"],
        1: ["place", "coffee"],
        2: ["pick", "box"],
        3: ["place", "box"],
        4: ["pick", "mug"],
        5: ["place", "mug"],
    }

    def __init__(
        self,
        task_keywords: dict[int, list[str]] | None = None,
        default_expert: int = 6,
    ):
        """
        Args:
            task_keywords: Expert ID → キーワードリストのマッピング。
                None の場合はデフォルトの公開6タスク定義を使用。
            default_expert: マッチしない場合のフォールバック Expert ID。
        """
        self.task_keywords = task_keywords or self.DEFAULT_TASK_KEYWORDS
        self.default_expert = default_expert

    def route(self, instruction: str) -> int:
        """キーワードマッチでタスクを分類し、Expert ID を返す。

        マッチング優先度: キーワード数が多いパターンを先にチェック
        （例: "pick coffee" が "pick" より優先）
        """
        instruction_lower = instruction.lower()

        # キーワード数が多い順にチェック（より具体的なパターンを優先）
        sorted_tasks = sorted(
            self.task_keywords.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )

        for expert_id, keywords in sorted_tasks:
            if all(kw in instruction_lower for kw in keywords):
                logger.info(
                    "Router: '%s' → Expert %d (keywords: %s)",
                    instruction,
                    expert_id,
                    keywords,
                )
                return expert_id

        logger.info(
            "Router: '%s' → Expert %d (default/general)",
            instruction,
            self.default_expert,
        )
        return self.default_expert
