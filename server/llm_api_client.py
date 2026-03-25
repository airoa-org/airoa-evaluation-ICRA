"""LLM PA Planner API クライアント

Docker 内から LLM API サーバー (別プロセス) に HTTP リクエストを送る。
LLMPAPlanner と同じインターフェース（decompose / replan）を提供。

Usage:
    client = LLMAPIClient(host="localhost", port=8001)
    pa_list = client.decompose("Pick up the bottle from the table.")
"""

import json
import logging
import urllib.request
import urllib.error
from typing import List, Optional

logger = logging.getLogger(__name__)


class LLMAPIClient:
    """LLM PA Planner API クライアント。

    LLMPAPlanner と同じ decompose() / replan() インターフェースを持つが、
    実際の推論は HTTP API 経由で別プロセスの LLM サーバーに委譲する。
    """

    def __init__(self, host: str = "localhost", port: int = 8001,
                 pa_map: Optional[dict] = None, timeout: float = 30.0):
        self._base_url = f"http://{host}:{port}"
        self._pa_map = pa_map or {}
        self._timeout = timeout
        self._available = None  # None = 未確認

    def is_available(self) -> bool:
        """LLM API サーバーが利用可能か確認。"""
        if self._available is not None:
            return self._available
        try:
            self._request("GET", "/health")
            self._available = True
            logger.info("LLM API server available at %s", self._base_url)
        except Exception:
            self._available = False
            logger.warning("LLM API server not available at %s", self._base_url)
        return self._available

    def decompose(self, sht: str, images=None) -> List[str]:
        """SHT → PA 列に分解する。

        1. ルール完全一致 / ファジーマッチ（ローカル）
        2. LLM API（リモート）
        3. E2E フォールバック
        """
        # ルール完全一致
        if sht in self._pa_map:
            return self._pa_map[sht]

        # ファジーマッチ
        sht_norm = sht.lower().strip().rstrip(".")
        for key, pa_list in self._pa_map.items():
            if key.lower().strip().rstrip(".") == sht_norm:
                return pa_list

        # LLM API
        if self.is_available():
            try:
                result = self._request("POST", "/decompose", {"sht": sht, "mode": "decompose"})
                pa_list = result.get("pa_list", [])
                if pa_list:
                    logger.info("LLM API decompose: '%s' -> %d PAs", sht[:50], len(pa_list))
                    return pa_list
            except Exception as e:
                logger.warning("LLM API decompose failed: %s", e)

        # E2E フォールバック
        logger.warning("LLM API unavailable, E2E fallback: '%s'", sht[:50])
        return [sht]

    def replan(self, sht: str, completed_pas: list, failed_pa: str,
               failure_reason: str, remaining_pas: list) -> List[str]:
        """PA 失敗時に残り PA 列を再計画する。"""
        if not self.is_available():
            return remaining_pas

        try:
            result = self._request("POST", "/decompose", {
                "sht": sht,
                "mode": "replan",
                "completed_pas": completed_pas,
                "failed_pa": failed_pa,
                "failure_reason": failure_reason,
                "remaining_pas": remaining_pas,
            })
            return result.get("pa_list", remaining_pas)
        except Exception as e:
            logger.warning("LLM API replan failed: %s", e)
            return remaining_pas

    def _request(self, method: str, path: str, data: dict = None) -> dict:
        url = f"{self._base_url}{path}"
        if data is not None:
            body = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(url, data=body, method=method)
            req.add_header("Content-Type", "application/json")
        else:
            req = urllib.request.Request(url, method=method)

        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
