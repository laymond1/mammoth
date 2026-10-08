"""CODA control: original method with an explicitly frozen encoder."""

from models.coda_prompt import CodaPrompt
from models.low_rank_utils.factory import CHECKPOINTS


class CodaFrozen(CodaPrompt):
    NAME = 'coda-frozen'

    @staticmethod
    def get_parser(parser):
        parser = CodaPrompt.get_parser(parser)
        parser.set_defaults(coda_vit_size='tiny', n_epochs=20, batch_size=16)
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        if args.coda_checkpoint is None:
            args.coda_checkpoint = CHECKPOINTS[args.coda_vit_size]['file']
        super().__init__(backbone, loss, args, transform, dataset)
        self.net.feat.requires_grad_(False)
