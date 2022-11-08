"""Data utilities."""
import random
from typing import Callable, Optional, Sequence, Tuple, Union
import torch
from torch.utils.data import Dataset, Subset

class RolloverTensorDataset(Dataset[Tuple[torch.Tensor, ...]]):
    """Like `TensorDataset`, but rolls over when the requested length exceeds the actual length."""

    def __init__(self, length: int, *tensors: torch.Tensor):
        assert all(tensors[0].size(0) == t.size(0) for t in tensors), \
               "Size mismatch between tensors"
        self.length = length
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(t[index % len(t)] for t in self.tensors)

    def __len__(self):
        return self.length


class DatasetsDataset(Dataset[Tuple]):
    """Extract values from a `datasets.Dataset` into a tuple."""

    def __init__(self, dataset: 'datasets.Dataset', keys: Sequence):
        self.dataset = dataset
        self.keys = keys

    def __getitem__(self, index):
        item = self.dataset[index]
        return tuple(item[key] for key in self.keys)

    def __len__(self):
        return len(self.dataset)


def load_dataset_subset(dataset: Dataset, indices: Optional[Sequence[int]]=None,
                        max_size: Optional[int]=None, shuffle: bool=False) -> Dataset:
    """Get a Dataset subset."""
    if indices is None:
        indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    if max_size is not None:
        indices = indices[:max_size]
    return Subset(dataset, indices)


# NOTE: We're importing dataset dependencies within functions---rather than at the top level---to
#       allow for reduced runtime dependencies on systems that don't need all datasets.


def load_dataset_glue(tokenizer: Callable, config: str, split: str, ubatch_size: int,
                      tok_padding: Union[bool, str]=True) -> Dataset:
    """Create a GLUE dataset."""
    # pylint: disable=import-outside-toplevel
    import datasets
    # When doing inference in batches (ubatch_size > 1), each item (tokenized sentence) in a batch
    # must have the same length, which requires padding shorter sentences in the batch.
    # 'transform' only operates on single items, so we'd be forced to use padding='max_length',
    # which always forces very long tensors, resulting in slower inference.
    # 'map' operates on batches, which allows for per-batch padding optimization.
    # 'transform' runs on-the-fly during dataset iteration, 'map' runs in advance and caches data.
    def map_function(batch):
        """Tokenize sentences in microbatches."""
        # Using return_tensors='pt' requires splitting the tensors afterward.
        # Use a numpy array instead, which will be stacked into a single PyTorch tensor later.
        encoding = tokenizer(batch['sentence'], padding=tok_padding, truncation=True,
                             return_tensors='np')
        batch.update(encoding)
        return batch
    # This datasets.Dataset should be copmatible with a pytorch Dataset
    dataset = datasets.load_dataset('glue', name=config, split=split)
    dataset = dataset.map(function=map_function, batched=True, batch_size=ubatch_size,
                          remove_columns=['sentence'])
    dataset.set_format(type='torch')
    return DatasetsDataset(dataset, ['input_ids', 'label'])


def load_dataset_imagenet(feature_extractor: Callable, root: str, split: str='train') -> Dataset:
    """Get the ImageNet dataset."""
    # pylint: disable=import-outside-toplevel
    from torchvision.datasets import ImageNet
    def transform(img):
        pixels = feature_extractor(images=img.convert('RGB'), return_tensors='pt')['pixel_values']
        # feature extractor expects a batch but we only have a single image, so drop the outer dim
        return pixels[0]
    return ImageNet(root, split=split, transform=transform)
