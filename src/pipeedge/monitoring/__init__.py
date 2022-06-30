"""Monitoring submodule."""
import csv
import dataclasses
import time
from typing import Any, Optional, Tuple, Union
import warnings
from apphb import logging, Heartbeat
from energymon.context import EnergyMon


_FIELD_TIME = None
_FIELD_WORK = 0
_FIELD_ENERGY = 1
_FIELD_ACCURACY = 2

def _heartbeat_factory(window_size=1, **kwargs):
    """Heartbeat factory function."""
    return Heartbeat(window_size, **kwargs)

def _heartbeat_create(window_size):
    """Create heartbeat with supported shapes."""
    # fields = work, energy, accuracy
    return _heartbeat_factory(window_size=window_size, time_shape=2, fields_shape=(1, 2, 1))

def _heartbeat_log_header(hbt, log_name, open_mode):
    if log_name is not None:
        with open(log_name, mode=open_mode, encoding="utf8") as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(logging.get_log_header(hbt,
                                                   time_name='Time (ns)',
                                                   heartrate_name='Heart Rate (/s)',
                                                   field_names=['Work',
                                                                'Energy (uJ)',
                                                                'Accuracy'],
                                                   field_rate_names=['Performance (/s)',
                                                                     'Power (W)',
                                                                     'Accuracy Rate (/s)']))

def _format_record(record):
    """Format float values to high precision, not in exponential form."""
    return [f'{r:.15f}' if isinstance(r, float) else r for r in record]


@dataclasses.dataclass
class MonitorIterationContext:
    """An iteration context - in general, clients should NOT modify this."""
    t_ns_last: Optional[int] = None
    e_uj_last: Optional[int] = None


@dataclasses.dataclass
class _HeartbeatContainer:
    """Class for heartbeat-related data."""
    hbt: Heartbeat = dataclasses.field(default_factory=_heartbeat_factory)
    log_name: Optional[str] = None
    log_mode: str = 'x'
    iter_ctx: MonitorIterationContext = dataclasses.field(default_factory=MonitorIterationContext)
    tag: int = 0


class MonitorContext:
    """
    The top-level monitoring interface.

    At its core, a monitor uses a ``Heartbeat`` to track timing/heartrate, work/performance,
    energy/power, and accuracy.

    Monitors use an ``EnergyMon`` native library to capture energy metrics.
    An energy reading's scope depends on the system and library capabilities.
    Energy readings are usually scoped to shared hardware components, e.g., CPU packages.
    Therefore, you should NOT assume that energy/power consumption can be ascribed exclusively to a
    particular monitor context or key.
    For example, iterations for multiple keys may be in-flight at any given time, or other system
    processes may also consume resources.

    This class is a reusable context manager.
    It is not reentrant.

    Parameters
    ----------
    key : Any, optional
        A unique key.
    window_size : int, optional
        The window period, where ``window_size > 0``.
        Ideally, ``window_size > 1``, otherwise `window` and `instant` values are the same.
    log_name : str, optional
        The log file to use.
    log_mode : str, optional
        The log file `open` mode to use - should be `w` or `x`.
        The value 'a' will also work if you don't mind multiple runtime records in one file.
    energy_lib : str, optional
        The EnergyMon library name, or `None` to disable energy monitoring.
    energy_lib_get : str, optional
        The EnergyMon library "getter" function.
    """
    # pylint: disable=too-many-public-methods

    def __init__(self, key: Any=None, window_size: int=1, log_name: Optional[str]=None,
                 log_mode: str='x', energy_lib: Optional[str]='energymon-default',
                 energy_lib_get: str='energymon_get_default'):
        # pylint: disable=too-many-arguments
        # define _initialized first so if hbt/energymon raise errors, __del__ will still see it
        self._initialized = False
        # remember the initial key - others may inherit from this (e.g., for window_size)
        self._key = key
        hbt = _heartbeat_create(window_size)
        self._hbt_ctxs = {
            key: _HeartbeatContainer(hbt, log_name, log_mode)
        }
        # EnergyMon requires a native library, so we make it optional for purely practical reasons.
        if energy_lib is None:
            self._em = None
        else:
            self._em = EnergyMon(lib=energy_lib, func_get=energy_lib_get)

    def keys(self) -> tuple:
        """Get a tuple of known keys."""
        return tuple(self._hbt_ctxs.keys())

    def add_heartbeat(self, key: Any=None, window_size: Optional[int]=None,
                      log_name: Optional[str]=None, log_mode: Optional[str]=None) -> None:
        """
        Add a heartbeat instance.

        Parameters
        ----------
        key : Any, optional
            A unique key.
        window_size : int, optional
            The window period.
            If `None`, will use the initialization key's window size.
        log_name : str, optional
            The log file to use.
        log_mode : str, optional
            The log file `open` mode to use - should be `w`, `x`, or `None`.
            The value 'a' will also work if you don't mind multiple runtime records in one file.
            If `None`, will use the initialization key's log mode.
        """
        if key in self._hbt_ctxs:
            raise ValueError(f'key already in use: {key}')
        if window_size is None:
            window_size = self.get_window_size(key=self._key)
        hbt = _heartbeat_create(window_size)
        if log_mode is None:
            log_mode = self._hbt_ctxs[self._key].log_mode
        self._hbt_ctxs[key] = _HeartbeatContainer(hbt, log_name, log_mode)
        if self._initialized:
            _heartbeat_log_header(hbt, log_name, self._hbt_ctxs[key].log_mode)

    def open(self) -> None:
        """Open the context."""
        if self._initialized:
            raise RuntimeError('Monitor is already open')
        if self._em is not None:
            self._em.init()
        self._initialized = True
        for hbtc in self._hbt_ctxs.values():
            _heartbeat_log_header(hbtc.hbt, hbtc.log_name, hbtc.log_mode)

    def close(self) -> None:
        """Close the context."""
        self._initialized = False
        if self._em is not None:
            self._em.finish()

    def _check_init(self):
        if not self._initialized:
            raise RuntimeError('Monitor is not open')

    def iteration_start(self, key: Any=None,
                        iter_ctx: Optional[MonitorIterationContext]=None) -> None:
        """
        Begin a measurement.

        Parameters
        ----------
        key : Any, optional
            A unique key, only used if `iter_ctx` is None.
        iter_ctx : MonitorIterationContext, optional
            The context for the new iteration.
            If not specified, uses the instance iteration context for `key`.
        """
        self._check_init()
        if iter_ctx is None:
            iter_ctx = self._hbt_ctxs[key].iter_ctx
        iter_ctx.t_ns_last = time.monotonic_ns()
        iter_ctx.e_uj_last = 0 if self._em is None else self._em.get_uj()

    def iteration(self, key: Any=None, work: int=1, accuracy: Union[int, float]=1,
                  iter_ctx: Optional[MonitorIterationContext]=None) -> None:
        """
        Complete a measurement.

        Parameters
        ----------
        key : Any, optional
            A unique key.
        work : int, optional
            The amount of work completed (e.g., number of application loops per monitor iteration).
        accuracy : Union[int, float], optional
            The measured or presumed accuracy for the work done in the iteration.
        iter_ctx : MonitorIterationContext, optional
            The context for the in-flight iteration.
            If not specified, uses the instance iteration context for `key`.
        """
        self._check_init()
        t_ns = time.monotonic_ns()
        e_uj = 0 if self._em is None else self._em.get_uj()
        hbtc = self._hbt_ctxs[key]
        if iter_ctx is None:
            iter_ctx = hbtc.iter_ctx
        # if the user calls this method before start, this is the start
        if iter_ctx.t_ns_last is not None:
            hbtc.hbt.heartbeat(hbtc.tag, (iter_ctx.t_ns_last, t_ns),
                               fields=((work,), (iter_ctx.e_uj_last, e_uj), (accuracy,)))
            hbtc.tag += 1
            if hbtc.log_name is not None:
                with open(hbtc.log_name, mode='a', encoding="utf8") as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                    # only normalize the rates
                    recs = logging.get_log_records(hbtc.hbt, count=1, heartrate_norm=1000000000,
                                                   field_rate_norms=[1000000000, 1000, 1000000000])
                    for rec in recs:
                        writer.writerow(_format_record(rec))
        iter_ctx.t_ns_last = t_ns
        iter_ctx.e_uj_last = e_uj

    def get_instant_time_s(self, key: Any=None) -> float:
        """Get instant time in seconds."""
        return self._hbt_ctxs[key].hbt.get_instant_count(fld=_FIELD_TIME) / 1000000000

    def get_instant_heartrate(self, key: Any=None) -> float:
        """Get instant heart rate (heartbeats/sec)."""
        return self._hbt_ctxs[key].hbt.get_instant_rate(fld=_FIELD_TIME) * 1000000000

    def get_instant_work(self, key: Any=None) -> int:
        """Get instant work."""
        return self._hbt_ctxs[key].hbt.get_instant_count(fld=_FIELD_WORK)

    def get_instant_perf(self, key: Any=None) -> float:
        """Get instant work rate (work/sec)."""
        return self._hbt_ctxs[key].hbt.get_instant_rate(fld=_FIELD_WORK) * 1000000000

    def get_instant_energy_j(self, key: Any=None) -> float:
        """Get instant energy in Joules."""
        return self._hbt_ctxs[key].hbt.get_instant_count(fld=_FIELD_ENERGY) / 1000000

    def get_instant_power_w(self, key: Any=None) -> float:
        """Get instant power in Watts."""
        return self._hbt_ctxs[key].hbt.get_instant_rate(fld=_FIELD_ENERGY) * 1000

    def get_instant_accuracy(self, key: Any=None) -> Union[int, float]:
        """Get instant accuracy."""
        return self._hbt_ctxs[key].hbt.get_instant_count(fld=_FIELD_ACCURACY)

    def get_instant_accuracy_rate(self, key: Any=None) -> float:
        """Get instant accuracy rate (acc/sec)."""
        return self._hbt_ctxs[key].hbt.get_instant_rate(fld=_FIELD_ACCURACY) * 1000000000

    def get_window_time_s(self, key: Any=None) -> float:
        """Get window time in seconds."""
        return self._hbt_ctxs[key].hbt.get_window_count(fld=_FIELD_TIME) / 1000000000

    def get_window_heartrate(self, key: Any=None) -> float:
        """Get window heart rate (heartbeats/sec)."""
        return self._hbt_ctxs[key].hbt.get_window_rate(fld=_FIELD_TIME) * 1000000000

    def get_window_work(self, key: Any=None) -> int:
        """Get window work."""
        return self._hbt_ctxs[key].hbt.get_window_count(fld=_FIELD_WORK)

    def get_window_perf(self, key: Any=None) -> float:
        """Get window work rate (work/sec)."""
        return self._hbt_ctxs[key].hbt.get_window_rate(fld=_FIELD_WORK) * 1000000000

    def get_window_energy_j(self, key: Any=None) -> float:
        """Get window energy in Joules."""
        return self._hbt_ctxs[key].hbt.get_window_count(fld=_FIELD_ENERGY) / 1000000

    def get_window_power_w(self, key: Any=None) -> float:
        """Get window power in Watts."""
        return self._hbt_ctxs[key].hbt.get_window_rate(fld=_FIELD_ENERGY) * 1000

    def get_window_accuracy(self, key: Any=None) -> Union[int, float]:
        """Get window accuracy."""
        return self._hbt_ctxs[key].hbt.get_window_count(fld=_FIELD_ACCURACY)

    def get_window_accuracy_rate(self, key: Any=None) -> float:
        """Get window accuracy rate (acc/sec)."""
        return self._hbt_ctxs[key].hbt.get_window_rate(fld=_FIELD_ACCURACY) * 1000000000

    def get_global_time_s(self, key: Any=None) -> float:
        """Get global time in seconds."""
        return self._hbt_ctxs[key].hbt.get_global_count(fld=_FIELD_TIME) / 1000000000

    def get_global_heartrate(self, key: Any=None) -> float:
        """Get global heart rate (heartbeats/sec)."""
        return self._hbt_ctxs[key].hbt.get_global_rate(fld=_FIELD_TIME) * 1000000000

    def get_global_work(self, key: Any=None) -> int:
        """Get global work."""
        return self._hbt_ctxs[key].hbt.get_global_count(fld=_FIELD_WORK)

    def get_global_perf(self, key: Any=None) -> float:
        """Get global work rate (work/sec)."""
        return self._hbt_ctxs[key].hbt.get_global_rate(fld=_FIELD_WORK) * 1000000000

    def get_global_energy_j(self, key: Any=None) -> float:
        """Get global energy in Joules."""
        return self._hbt_ctxs[key].hbt.get_global_count(fld=_FIELD_ENERGY) / 1000000

    def get_global_power_w(self, key: Any=None) -> float:
        """Get global power in Watts."""
        return self._hbt_ctxs[key].hbt.get_global_rate(fld=_FIELD_ENERGY) * 1000

    def get_global_accuracy(self, key: Any=None) -> Union[int, float]:
        """Get global accuracy."""
        return self._hbt_ctxs[key].hbt.get_global_count(fld=_FIELD_ACCURACY)

    def get_global_accuracy_rate(self, key: Any=None) -> float:
        """Get global accuracy rate (acc/sec)."""
        return self._hbt_ctxs[key].hbt.get_global_rate(fld=_FIELD_ACCURACY) * 1000000000

    def get_tag(self, key: Any=None) -> int:
        """The next tag."""
        return self._hbt_ctxs[key].tag

    def get_window_size(self, key: Any=None) -> int:
        """The window size."""
        return self._hbt_ctxs[key].hbt.window_size

    # Properties

    @property
    def initialized(self) -> bool:
        """Whether the Monitor is initialized (opened)."""
        return self._initialized

    @property
    def energy_source(self) -> str:
        """Human-readable description of the energy monitoring source."""
        return 'None' if self._em is None else self._em.get_source()

    # Context management

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # Safe cleanup

    def __del__(self):
        if self._initialized:
            warnings.warn('unclosed monitor', category=ResourceWarning, source=self)
            self.close()

    # Serialization

    def __getstate__(self):
        # This class holds state that cannot be (or doesn't make sense to) pickle
        raise TypeError(f"Cannot pickle {self.__class__.__name__!r} object")
