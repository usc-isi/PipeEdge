# pylint: disable=missing-function-docstring
"""Test comm.rpc.DistRpcContext."""
import os
import unittest
from edgepipe.comm.rpc import DistRpcContext

MASTER_ADDR = 'localhost'
MASTER_PORT = '29501'
NUM_WORKER_THREADS = 16


class TestDistRpcContext(unittest.TestCase):
    """Test DistRpcContext."""

    @classmethod
    def setUpClass(cls):
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT

    def test_lifecycle(self):
        # pylint: disable=no-self-use
        ctx = DistRpcContext(1, 0, NUM_WORKER_THREADS)
        ctx.init()
        ctx.shutdown()

    def test_context(self):
        # pylint: disable=no-self-use
        with DistRpcContext(1, 0, NUM_WORKER_THREADS):
            pass
