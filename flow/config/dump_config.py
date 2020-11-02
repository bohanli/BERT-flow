import json

model_config = {
    "latent_architecture": "glow_resnet",
    "activation": "relu", # Activation - Relu or Gatu
    "coupling": "affine", # Coupling layer, additive or affine.
    "coupling_width": 512,
    "coupling_dropout": 0.0,
    "top_prior": "normal",
    "n_levels": 3,
    "depth": 6,

    "use_fp16": False # not implemented yet
}

with open('/mnt/cephfs_new_wj/mlnlp/libohan.05/text_flow/config/model_config_normal.json', 'w') as jp:
    json.dump(model_config, jp, indent=4)

