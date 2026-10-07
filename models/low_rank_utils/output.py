"""Mammoth backbone return conventions for original source network results."""


def backbone_output(result, returnt=None):
    if returnt is None or returnt in ('full', 'all'):
        return result
    if returnt == 'out':
        return result['logits']
    if returnt == 'features':
        return result['features']
    if returnt == 'both':
        return result['logits'], result['features']
    raise ValueError(f'Unsupported LoRA backbone return type: {returnt}')
