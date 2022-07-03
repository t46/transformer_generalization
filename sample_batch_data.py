
import dataset
import torch
import framework
from typing import Optional, Dict, Any
from framework.utils import U

CFQ_SPLIT = 'mcd2'
BATCH_SIZE = 512
BATCH_DIM = 1
SEED = 0
TRAIN_NUM_WORKERS = 1
DEVICE = 'cuda'


def create_train_loader(loader: torch.utils.data.Dataset, seed: Optional[int] = None) \
                        -> torch.utils.data.DataLoader:

    return torch.utils.data.DataLoader(loader, batch_size=BATCH_SIZE,
                                        sampler=framework.loader.sampler.InfiniteSampler(
                                            loader, seed=SEED),
                                        collate_fn=framework.loader.collate.VarLengthCollate(
                                            batch_dim=BATCH_DIM),
                                        num_workers=TRAIN_NUM_WORKERS, pin_memory=True)


def get_train_batch(data_iter) -> Dict[str, Any]:
    return next(data_iter)


def to_device(data: Any) -> Any:
    return U.apply_to_tensors(data, lambda d: d.to(DEVICE))


def prepare_data(data: Dict[str, Any]) -> Dict[str, Any]:
    return to_device(data)


train_set = dataset.CFQ(["train"], split_type=[CFQ_SPLIT])
valid_sets_val = dataset.CFQ(["val"], split_type=[CFQ_SPLIT])
valid_sets_test = dataset.CFQ(["test"], split_type=[CFQ_SPLIT])

train_loader = create_train_loader(train_set)
data_iter = iter(train_loader)

data = prepare_data(get_train_batch(data_iter))
print(data)
