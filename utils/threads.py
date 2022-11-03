"""Thread utilities."""
from contextlib import contextmanager
import threading

class RWLock:
    """A multiple-reader, non-reentrant single-writer lock."""

    def __init__(self):
        self._readers = 0
        # Use a simple Lock, not Condition's default RLock
        self._cond = threading.Condition(threading.Lock())

    def acquire_read(self):
        """Acquire a read lock."""
        with self._cond:
            self._readers += 1

    def release_read(self):
        """Release a read lock."""
        with self._cond:
            self._readers -= 1
            self._cond.notifyAll()

    def acquire_write(self):
        """Acquire a write lock."""
        self._cond.acquire()
        while self._readers > 0:
            self._cond.wait()

    def release_write(self):
        """Release a write lock."""
        self._cond.release()

    @contextmanager
    def lock_read(self):
        """A context manager factory function to acquire a read lock."""
        try:
            self.acquire_read()
            yield
        finally:
            self.release_read()

    @contextmanager
    def lock_write(self):
        """A context manager factory function to acquire a write lock."""
        try:
            self.acquire_write()
            yield
        finally:
            self.release_write()

    # Class instance context acquire/release conservatively uses writer lock

    __enter__ = acquire_write

    def __exit__(self, _t, _v, _tb):
        self.release_write()


class ThreadSafeCounter:
    """Thread-safe counter."""

    def __init__(self, value: int=0):
        self._value = value
        self._cond = threading.Condition()

    @property
    def value(self) -> int:
        """Current counter value."""
        with self._cond:
            val = self._value
            self._cond.notify_all()
        return val

    def add(self, quantity: int=1) -> None:
        """Add to counter atomically."""
        with self._cond:
            self._value += quantity
            self._cond.notify_all()

    def set(self, value: int=0) -> None:
        """Set (or reset) counter value."""
        with self._cond:
            self._value = value
            self._cond.notify_all()

    def wait_gte(self, threshold: int) -> None:
        """Wait until counter >= threshold."""
        with self._cond:
            while self._value < threshold:
                self._cond.wait()
