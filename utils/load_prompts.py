import json
from typing import List, Any, Generator


def load_prompts(configs):
    prompts = []
    if configs.text:
        prompts.extend([{"idx": -1, "prompt": configs.text}] * configs.num)
    if configs.path:
        f = open(configs.path, "r")
        for idx, line in enumerate(f.readlines()):
            line = json.loads(line)
            line["idx"] = idx
            prompts.extend([line] * configs.num)
    return prompts

def prompt_loader(
    configs
) -> Generator[Any, None, None]:
    prompts = load_prompts(configs)
    for i in range(0, len(prompts), configs.batch_size):
        end_idx = i + configs.batch_size
        batch = prompts[i:end_idx]
        yield batch