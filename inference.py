import json
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


def validate_on(set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader, model, in_vocabulary, out_vacabulary) -> Tuple[Any, float, Dict]:
    MODEL_PATH = '/root/projects/transformer_generalization/save/cfq_mcd_small_batch_universal/checkpoint/model-50000.pth'
    LABEL_SMOOTHING = 0.0

    model.load_state_dict(torch.load(MODEL_PATH)['model'])
    model.eval()
    model_interface = create_model_interface(model, LABEL_SMOOTHING)

    result = {'results': []}

    with torch.no_grad():
        loss_sum = 0

        test = set.start_test()
        for d in tqdm(loader):
            d = prepare_data(d)
            res = model_interface(d)
            digits = model_interface.decode_outputs(res)  # outputs, output_lengths
            loss_sum += res.loss.sum().item() * res.batch_size

            predicted_ids_all_batch = torch.argmax(digits[0][:-1, :, : -1], dim=-1)  # TODO: labelよりtokenの方がbatchsize以外の次元が１大きい
            label_ids_all_batch = d['out']
            input_ids_all_batch = d['in']
            # TODO: predictionとlabelはIDが0-96まであるが、out_vacaburary.inv_wordsは0から95まで。
            # TODO: おそらく、labelの0はpaddingなので、predictionを1-95までにするのが良い気はする。
            # TODO: 一方で、両方ゼロ始まりの場合最初の数サンプルの予測結果がとても合っているので、predictionだけ１始まりにするように
            # TODO: ずらすのは何か違う感じがする。
            for batch_id in range(predicted_ids_all_batch.shape[1]):
                predicted_tokens = [out_vacabulary.inv_words[int(token_id)] for token_id in predicted_ids_all_batch[:, batch_id]]
                label_tokens = [out_vacabulary.inv_words[int(label_id)] for label_id in label_ids_all_batch[:, batch_id]]
                input_tokens = [in_vocabulary.inv_words[int(input_id)] for input_id in input_ids_all_batch[:, batch_id]]
                result['results'].append(
                    {
                        'input_sentence': ' '.join(input_tokens),
                        'predicted_sentence': ' '.join(predicted_tokens),
                        'label_sentence': ' '.join(label_tokens),
                        # 'predicted_tokens': predicted_tokens,
                        # 'label_tokens': label_tokens,
                        'exact_match': int(torch.equal(predicted_ids_all_batch[:, batch_id], label_ids_all_batch[:, batch_id]))
                    }
                )

            test.step(digits, d)

    return test, loss_sum / len(set), result


def create_valid_loader(vset: torch.utils.data.Dataset, BATCH_SIZE, batch_dim, VALID_NUM_WORKERS) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(vset, batch_size=BATCH_SIZE,
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
    BATCH_SIZE = 512
    VALID_NUM_WORKERS = 0
    BATCH_DIM = 1
    SPLIT = 'val'  # 'train', ’val' or 'test'  TODO: 'train'は厳密に正しいかはわからない。性能を見るに良さそう。
    SAVE_PATH = f'/root/projects/transformer_generalization/results/cfq_mcd_small_batch_universal_{SPLIT}.json'
    data_set = dataset.CFQ([SPLIT], split_type=[CFQ_SPLIT])
    data_loader = create_valid_loader(data_set, BATCH_SIZE, BATCH_DIM, VALID_NUM_WORKERS)
    model = create_model(len(data_set.in_vocabulary), len(data_set.out_vocabulary)).to('cuda')
    valid, loss, result = validate_on(data_set, data_loader, model, data_set.in_vocabulary, data_set.out_vocabulary)
    accuracy = valid.accuracy
    print(f'loss: {loss}, accuracy: {accuracy}')
    with open(SAVE_PATH, 'w') as f:
        json.dump(result, f)
