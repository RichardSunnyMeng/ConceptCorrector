import os
import random
import torch

from PIL import Image
from llava.model.builder import load_pretrained_model

from checker.llava_next import llava_forward


PROMPT_TEMPLATE = \
f'''
Check whether the following elements exist in this picture respectively. 
For each element, give the answer (Y for exist, N for non-exist, no other options) in "[ ]".
[PLACEHOLDER_FOR_CONCEPTS]

An example:
[PLACEHOLDER_FOR_EXAMPLES]
This example has nothing to do with the answer, it only provides the format for the answer.
'''


class DiffusionChecker(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.load_model(config)
        self.load_concepts(config)
        self.load_prompts(config)
        self.cache = config.checker_cache

    def load_model(self, config):
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=config.model.model_path,
            model_base=None,
            model_name=config.model.model_name,
            device_map=config.model.device_map
        )
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.model.tie_weights()
    
    def load_concepts(self, config):
        self.concepts_to_negatives = vars(config.concepts)

    def load_prompts(self, config):
        concepts_strlist = ""
        for concept in self.concepts_to_negatives.keys():
            concepts_strlist += f"- {concept}: [ ]\n"
        self.prompt = PROMPT_TEMPLATE.replace("[PLACEHOLDER_FOR_CONCEPTS]", concepts_strlist)
        self.prompt = self.prompt.replace("[PLACEHOLDER_FOR_EXAMPLES]", concepts_strlist.replace("[ ]", "[N]"))
    
    def query_model(self, image):
        response = llava_forward(
            tokenizer=self.tokenizer,
            model=self.model,
            image_processor=self.image_processor,
            model_name="llava-onevision-qwen2-0.5b-si",
            query=self.prompt,
            image_files=self.cache,
        )
        return self.process_response(response)

    def process_response(self, response):
        yes_concepts = []
        neg_concepts = []
        response_lines = response.split("\n")
        for line in response_lines:
            for concept in self.concepts_to_negatives.keys():
                if concept in line:
                    if "[Y]" in line or "[yes]" in line or "Y" in line or "yes" in line:
                        yes_concepts.append(concept)
                        neg_concepts.append(self.concepts_to_negatives[concept])
                    break
        return ", ".join(yes_concepts), ", ".join(neg_concepts)
    
    def check(self, images: list[Image]):
        random_state_cpu = torch.get_rng_state()
        random_state_cuda = torch.cuda.get_rng_state_all()

        torch.manual_seed(2020)
        torch.cuda.manual_seed_all(2020)

        results = []
        for image in images:
            image.save(self.cache)
            result = self.query_model(image)
            results.append(result)

        torch.set_rng_state(random_state_cpu)
        torch.cuda.set_rng_state_all(random_state_cuda)

        return results
    
    def get_negative_concept(self, concept: str):
        return self.concepts_to_negatives[concept]

