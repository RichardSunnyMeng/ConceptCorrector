import os
import sys
import json

import torch

from checker import DiffusionChecker
from modified_diffusion import load_diffusion_pipe
from utils import load_yaml_as_argparse, prompt_loader

if __name__ == "__main__":
    configs = load_yaml_as_argparse("configs.yaml")

    # hyper
    save_path = configs.generation.save_path
    seed = configs.generation.hyper.seed
    guidance_scale = configs.generation.hyper.guidance_scale
    num_inference_steps = configs.generation.hyper.num_inference_steps
    checkpoints = configs.generation.hyper.checkpoints

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    concept_checker = DiffusionChecker(configs.checker)
    diffuser_pipe = load_diffusion_pipe(configs.generation.model)

    os.makedirs(save_path, exist_ok=True)
    result_log = open(save_path + "/results.json", "w")

    for batch in prompt_loader(configs.generation.prompts):
        idx = [x["idx"] for x in batch]
        prompts = [x["prompt"] for x in batch]

        if "sd_seed" in batch[0].keys():
            seed = batch[0]["sd_seed"]
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        if "sd_guidance_scale" in batch[0].keys():
            guidance_scale = batch[0]["sd_guidance_scale"] 

        out = diffuser_pipe(prompts, 
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale, 
            concept_checker=concept_checker,
            checkpoints=checkpoints,
        )

        for num, image in enumerate(out.images):
            image.save(os.path.join(save_path, f'{idx[num]}_{num}.png'))
        
            log = {"img_path": os.path.join(save_path, f'{idx[num]}_{num}.png')}
            log["check_results"] = {}
            for checkpoint in checkpoints:
                log["check_results"][checkpoint] = out.check_results[checkpoint][num]

            result_log.write(json.dumps(log))
            result_log.write("\n")
            result_log.flush()
        
    