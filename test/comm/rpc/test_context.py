# pylint: disable=missing-function-docstring
"""Test comm.rpc.DistRpcContext."""
import os
import unittest
from pipeedge.comm.rpc import DistRpcContext

MASTER_ADDR = 'localhost'
MASTER_PORT = '29501'

INIT_ARGS = ('worker0',)
INIT_KWARGS = { 'world_size': 1, 'rank': 0 }

class TestDistRpcContext(unittest.TestCase):
    """Test DistRpcContext."""

    @classmethod
    def setUpClass(cls):
        os.environ['MASTER_ADDR'] = MASTER_ADDR
        os.environ['MASTER_PORT'] = MASTER_PORT

    def test_lifecycle(self):
        # pylint: disable=no-self-use
        ctx = DistRpcContext(INIT_ARGS, INIT_KWARGS)
        ctx.init()
        ctx.shutdown()

    def test_context(self):
        # pylint: disable=no-self-use
        with DistRpcContext(INIT_ARGS, INIT_KWARGS):
            pass
