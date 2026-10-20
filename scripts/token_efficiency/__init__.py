"""Token-reduction efficiency study: one torch-only package that runs unchanged on every device.

The package imports nothing from the rest of the repository, so it can be copied to a
Jetson or a Raspberry Pi as a plain ``token_efficiency`` folder. ``reducers.py`` is a
verbatim copy of ``backbone/token_compression/reducers.py`` on branch ``tocom/plugin``
(commit bf960df); ``VENDORED`` pins it.
"""

VENDORED = {
    'reducers.py': {
        'source': 'tocom/plugin:backbone/token_compression/reducers.py@bf960df',
        'sha256': '65cc82101b3efed14e6e5d0731f98b40293f5459a17037b91dff8950d9402732',
    },
}
