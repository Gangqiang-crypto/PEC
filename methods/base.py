"""统一方法接口 BaseTopKMethod。"""

from abc import ABC, abstractmethod
from typing import Any

from common.metrics import Recorder, QueryResult
from common.shared_index import SharedEncryptedIndex
from common.token import QueryToken


class BaseTopKMethod(ABC):
    name: str = "base"
    display_name: str = "base"
    architecture: str = "single_server"

    def prepare(self, shared_index: SharedEncryptedIndex, config: Any, **kwargs):
        pass

    @abstractmethod
    def query(
        self,
        token: QueryToken,
        shared_index: SharedEncryptedIndex,
        recorder: Recorder,
    ) -> QueryResult:
        raise NotImplementedError

    def estimate_comm(
        self,
        recorder: Recorder,
        token: QueryToken,
        config: Any,
    ):
        """根据recorder中的计数器和config估算通信量，写入recorder.comm。"""
        pass