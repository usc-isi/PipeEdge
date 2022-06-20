"""Communication utilities."""
import threading


class DistRequestWaitDaemon(threading.Thread):
    """
    Thread for waiting on asynchronous distributed requests.

    Parent thread should poll is_alive() to determine if the request is complete.
    """
    # This is a hack to get around is_completed() not working.
    # Using wait() blocks forever, which prevents normal threads from stopping cleanly on command.
    # See: https://github.com/pytorch/pytorch/issues/30723

    def __init__(self, req):
        super().__init__(daemon=True)
        self._req = req

    def run(self):
        """Wait for request."""
        self._req.wait()
