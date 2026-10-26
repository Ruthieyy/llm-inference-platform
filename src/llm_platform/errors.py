class QueueFullError(Exception):
    """The admission queue cannot accept another request."""


class UpstreamError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
