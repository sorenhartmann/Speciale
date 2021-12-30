from typing import Tuple

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor

BATCH_IN = Tuple[Tensor, Tensor]


class InferenceModule(pl.LightningModule):
    ...

    def training_step(self, batch: BATCH_IN, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        pass
