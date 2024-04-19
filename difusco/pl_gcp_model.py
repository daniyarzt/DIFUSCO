

from pl_meta_model import COMetaModel


class GCPModel(COMetaModel):
    def __init__(self, param_args=None):
        raise NotImplementedError()

    def forward(self, x, t, edge_index):
        raise NotImplementedError()

    def categorical_training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def gaussian_training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
        raise NotImplementedError()

    def gaussian_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx, draw=False, split='test'):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()
