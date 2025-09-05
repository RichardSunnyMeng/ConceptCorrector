import os
import json
import torch
from diffusers import DDIMScheduler
from modified_diffusion.modified_unet import UNet2DConditionModifiedModel
from modified_diffusion.modified_stable_diffusion_pipeline import StableDiffusionWithCheckpointPipeline


def modify_unet(configs, loaded_pipe):
    config_path = os.path.join(configs.model_path, "unet", "config.json")
    config = json.load(open(config_path, "r"))
    
    erased_keys = []
    for k in config.keys():
        if k.startswith("_"):
            erased_keys.append(k)
    
    for k in erased_keys:
        config.pop(k)
    unet = UNet2DConditionModifiedModel(**config).to(loaded_pipe.unet.device, loaded_pipe.dtype)
    unet.load_state_dict(loaded_pipe.unet.state_dict())

    del loaded_pipe.unet
    loaded_pipe.unet = unet

def modify_attn_processor(loaded_pipe):
    xattn_list1 = [
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2",

        "up_blocks.1.attentions.0.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2",
    ]

    xattn_list2 = [
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2",
        "mid_block.attentions.0.transformer_blocks.0.attn2",
    ]

    from modified_diffusion.modified_attention_processor \
        import RemovalAttnProcessorWithCalMasks, RemovalAttnProcessorWithReadMasks

    for name, module in loaded_pipe.unet.named_modules():
        if name in xattn_list1:
            module.set_processor(RemovalAttnProcessorWithCalMasks())
        if name in xattn_list2:
            module.set_processor(RemovalAttnProcessorWithReadMasks())


def load_diffusion_pipe(configs):
    pipe = StableDiffusionWithCheckpointPipeline.from_pretrained(
            configs.model_path, 
            torch_dtype=torch.float16,
        ).to(configs.model_device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if not configs.enable_safety_checker:
        pipe.safety_checker = None
    
    modify_unet(configs, pipe)
    modify_attn_processor(pipe)
    return pipe