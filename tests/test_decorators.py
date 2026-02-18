import logging
from src.utils.decorators import log_call, timer

class Dummy:
    def __init__(self) -> None:
        self.logger = logging.getLogger("dummy")
        self.logger.addHandler(logging.NullHandler())

    @log_call
    @timer
    def compute(self, x: int) -> int:
        return x * 2

def test_decorators_execute_without_error() -> None:
    obj = Dummy()
    result = obj.compute(5)
    assert result == 10