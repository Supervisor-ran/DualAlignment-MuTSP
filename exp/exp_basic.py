import os
import torch
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
    Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, DForecaster, NLinear, MaTransformer, UniTime, TimeLLM, iTransformer_plus, PatchTST_plus, NLinear_plus, \
    iTransformer_plus_original, MaTransformer_DA, UniTime_DA, TimeLLM_DA, DATransformer, DATransformer_test


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'LightTS': LightTS,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'MICN': MICN,
            'Crossformer': Crossformer,
            'FiLM': FiLM,
            'iTransformer': iTransformer,
            'Koopa': Koopa,
            'TiDE': TiDE,
            'FreTS': FreTS,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'SegRNN': SegRNN,
            'DForecaster': DForecaster,
            'NLinear': NLinear,
            'MaTransformer': MaTransformer,
            'UniTime': UniTime,
            'TimeLLM': TimeLLM,
            'iTransformer_plus': iTransformer_plus,
            "PatchTST_plus": PatchTST_plus,
            "NLinear_plus": NLinear_plus,
            "iTransformer_plus_original": iTransformer_plus_original,
            "MaTransformer_DA": MaTransformer_DA,
            "UniTime_DA": UniTime_DA,
            "TimeLLM_DA": TimeLLM_DA,
            "DATransformer": DATransformer,
            "DATransformer_test": DATransformer_test,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
