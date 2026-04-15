import abc


class BaseContentSplitter(abc.ABC):
    @abc.abstractmethod
    def extract(self, text: str) -> str: ...
