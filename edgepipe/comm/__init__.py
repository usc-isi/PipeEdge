"""Communication module."""

class DistContext():
    """Parent class for distributed context managers."""

    def __init__(self, init_args: tuple, init_kwargs: dict):
        self._init_args = init_args
        self._init_kwargs = init_kwargs
        self._world_size = init_kwargs['world_size']
        self._rank = init_kwargs['rank']
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
