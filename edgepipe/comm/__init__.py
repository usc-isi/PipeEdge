"""Communication module."""

class DistContext():
    """Parent class for distributed context managers."""

    def __init__(self, world_size, rank):
        self._world_size = world_size
        self._rank = rank
        self._initialized = False

    def init(self):
        """Initialize the distributed context."""
        self._initialized = True

    def shutdown(self):
        """Shutdown the distributed context."""
        self._initialized = False

    def __enter__(self):
        assert not self._initialized
        self.init()
        return self

    def __exit__(self, *args):
        assert self._initialized
        self.shutdown()
