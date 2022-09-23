"""Communication utilities."""
import io
import pickle
import threading
import torch


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


# Based on: torch.distributed.distributed_c10d.py:_object_to_tensor
def object_to_tensor(obj, device):
    """Convert a Python object to a `torch.Tensor`."""
    bytes_io = io.BytesIO()
    pickle.Pickler(bytes_io).dump(obj)
    byte_storage = torch.ByteStorage.from_buffer(bytes_io.getvalue())  # type: ignore[attr-defined]
    # Do not replace `torch.ByteTensor` or `torch.LongTensor` with torch.tensor and specifying dtype.
    # Otherwise, it will casue 100X slowdown.
    # See: https://github.com/pytorch/pytorch/issues/65696
    byte_tensor = torch.ByteTensor(byte_storage).to(device)
    local_size = torch.LongTensor([byte_tensor.numel()]).to(device)
    return byte_tensor, local_size


# Based on: torch.distributed.distributed_c10d.py:_tensor_to_object
def tensor_to_object(tensor, tensor_size):
    """Convert a `torch.Tensor` to a Python object."""
    tensor = tensor.cpu()
    buf = tensor.numpy().tobytes()[:tensor_size]
    return pickle.Unpickler(io.BytesIO(buf)).load()
