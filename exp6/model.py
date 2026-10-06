"""Reuse the repository's exact ViT-Tiny implementation and local weight mapping."""
from exp6.runtime import configure
from exp2.model import pretrained_vit, initialization_digest, checkpoint_sha256
# exp2.model sets its own cache environment on import. Reset before any model creation.
configure()
from exp6.lora import inject


def create_model():
    model, metadata = pretrained_vit()
    layers = inject(model)
    metadata['initialization_sha256'] = initialization_digest(model)
    metadata['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata['frozen_parameters'] = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    metadata['lora_modules'] = list(layers)
    return model, metadata
