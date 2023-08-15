# Equiareal batch sampler

Standard practice in Deep Learning is to train models on batches of data, keeping the number of samples in a batch ("batch size") constant.
However, what you really need is a constant memory footprint of a batch, to i.e. coordinate it with the memory of your GPU.
If your samples have different sizes (common in text, time series data), constant batch size will lead to a highly variable memory footprint.
This package provides a batch sampler that keeps constant "batch area": sum of lengths of samples.
Batch area mostly corresponds to its memory footprint, although padding will increase it slightly.

## Installation

```
pip install equibatch
```

## Usage


```python
from equibatch import EquiarealBatchSampler

data = [
    'London',
    'Birmingham',
    'Glasgow',
    'Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch',
    'Liverpool',
    'Bristol',
    'Manchester'
]

batch_sampler = EquiarealBatchSampler(
    sampler = range(len(data)), # the order in which the dataset is traversed
    len_checker = lambda ix: len(data[ix]), # definition of "footprint of a sample"
    max_size = 10, # maximum number of samples in a batch
    max_footprint = 60 # maximum cumulative footprint of a batch
    )

for batch in batch_sampler:
    sample = [data[ix] for ix in batch] 
    print(sample)
```

This will print

```
['London', 'Birmingham', 'Glasgow']
['Llanfairpwllgwyngyllgogerychwyrndrobwllllantysiliogogogoch']
['Liverpool', 'Bristol', 'Manchester']
```

## Pytorch support

If you have Pytorch installed, `EquiarealBatchSampler` will be a subclass of `torch.utils.data.sampler.BatchSampler` and you can use it in your `torch.utils.data.DataLoader` like this:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset = data,
    batch_sampler = batch_sampler
)

for input in dataloader:
    output = model(input)
    loss = loss_fn(output)
    loss.backward()
    optimizer.step()
```