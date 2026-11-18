"""Model registry. Each mode maps to swappable model components.

Swapping a model (e.g. for a commercially-licensed build) only touches this file + its wrapper.
"""
MODES = {
    "pro": {
        "label": "Pro (best quality)",
        "segmenter": "sam2.1-hiera-base-plus",   # Apache-2.0
        "matter": "matanyone",                    # S-Lab non-commercial
    },
    "fast": {
        "label": "Fast (people)",
        "matter": "rvm-resnet50",                 # GPL-3.0
    },
    "detail": {
        "label": "Detail (fine edges)",
        "matter": "birefnet-hr",                  # MIT
    },
}
