import torch
import dataset
import framework
from typing import Tuple, Dict, Any
from tqdm import tqdm
from framework.utils import U
from interfaces import TransformerEncDecInterface
from layers.transformer import Transformer, UniversalTransformer, RelativeTransformer, UniversalRelativeTransformer
from models import TransformerEncDecModel


ARGS = {
    "transformer.variant": "relative_universal",
    "state_size": 256,
    "transformer.n_heads": 4,
    "transformer.ff_multiplier": 2,
    "transformer.encoder_n_layers": 6,
    "transformer.decoder_n_layers": 6,
    "tranformer.tied_embedding": True
       }


def to_device(data: Any, device: str) -> Any:
    return U.apply_to_tensors(data, lambda d: d.to(device))


def prepare_data(data: Dict[str, Any]) -> Dict[str, Any]:
    return to_device(data, 'cuda')


def create_model_interface(model, label_smoothing):
    model_interface = TransformerEncDecInterface(model, label_smoothing)
    return model_interface


def validate_on(set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader, model) -> Tuple[Any, float]:
    MODEL_PATH = '/root/projects/transformer_generalization/save/cfq_mcd_small_batch_universal/checkpoint/model-50000.pth'
    LABEL_SMOOTHING = 0.0

    model.load_state_dict(torch.load(MODEL_PATH)['model'])
    model.eval()
    model_interface = create_model_interface(model, LABEL_SMOOTHING)

    with torch.no_grad():
        loss_sum = 0

        test = set.start_test()
        for d in tqdm(loader):
            d = prepare_data(d)
            res = model_interface(d)
            digits = model_interface.decode_outputs(res)
            loss_sum += res.loss.sum().item() * res.batch_size

            test.step(digits, d)

    return test, loss_sum / len(set)


def create_valid_loader(vset: torch.utils.data.Dataset, test_batch_size, batch_dim, VALID_NUM_WORKERS) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(vset, batch_size=test_batch_size,
                                       collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=batch_dim),
                                       num_workers=VALID_NUM_WORKERS)


def create_model(in_voc_size, out_voc_size) -> torch.nn.Module:
    rel_args = dict(pos_embeddig=(lambda x, offset: x), embedding_init="xavier")
    trafos = {
        "scaledinit": (Transformer, dict(embedding_init="kaiming", scale_mode="down")),
        "opennmt": (Transformer, dict(embedding_init="xavier", scale_mode="opennmt")),
        "noscale": (Transformer, {}),
        "universal_noscale": (UniversalTransformer, {}),
        "universal_scaledinit": (UniversalTransformer, dict(embedding_init="kaiming", scale_mode="down")),
        "universal_opennmt": (UniversalTransformer, dict(embedding_init="xavier", scale_mode="opennmt")),
        "relative": (RelativeTransformer, rel_args),
        "relative_universal": (UniversalRelativeTransformer, rel_args)
    }

    constructor, args = trafos[ARGS["transformer.variant"]]

    return TransformerEncDecModel(in_voc_size,
                                  out_voc_size,
                                  ARGS["state_size"],
                                  nhead=ARGS["transformer.n_heads"],
                                  num_encoder_layers=ARGS["transformer.encoder_n_layers"],
                                  num_decoder_layers=ARGS["transformer.decoder_n_layers"] or \
                                                     ARGS["transformer.encoder_n_layers"],
                                  ff_multiplier=ARGS["transformer.ff_multiplier"],
                                  transformer=constructor,
                                  tied_embedding=ARGS["tranformer.tied_embedding"], **args)


if __name__ == "__main__":
    CFQ_SPLIT = 'mcd2'
    TEST_BATCH_SIZE = 512
    VALID_NUM_WORKERS = 0
    BATCH_DIM = 1
    test_set = dataset.CFQ(["test"], split_type=[CFQ_SPLIT])
    test_loader = create_valid_loader(test_set, TEST_BATCH_SIZE, BATCH_DIM, VALID_NUM_WORKERS)
    model = create_model(len(test_set.in_vocabulary), len(test_set.out_vocabulary)).to('cuda')
    validate_on(test_set, test_loader, model)
