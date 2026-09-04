from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

ModelT = TypeVar("ModelT")
ResultT = TypeVar("ResultT")


@dataclass
class BatchModelExecutor(Generic[ModelT, ResultT]):
    loader: Callable[[str], ModelT]
    processor: Callable[[ModelT, str], ResultT]
    releaser: Callable[[ModelT], None]

    def run(self, model_id: str, problems: list[str]) -> list[ResultT]:
        model = self.loader(model_id)
        try:
            return [self.processor(model, problem) for problem in problems]
        finally:
            self.releaser(model)
            del model
            gc.collect()

