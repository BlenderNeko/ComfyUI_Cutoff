import torch
import os
import sys
import copy
import re
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

def replace_embeddings(max_token, prompt, replacements=None):
    
    if replacements is None:
        emb_lookup = []
    else:
        emb_lookup = replacements.copy()
        max_token += len(emb_lookup)

    def get_replacement(embedding):
        for e, n in emb_lookup:
            if torch.equal(embedding, e):
                return n
        return None
    
    tokens = []
    for x in prompt:
        row = []
        for i in range(len(x)):
            emb = x[i][0]
            if not torch.is_tensor(emb):
                row.append(emb)
            else:
                n = get_replacement(emb)
                if n is not None:
                    row.append(n)
                else:
                    max_token += 1
                    row.append(max_token)
                    emb_lookup.append((emb,max_token))
        tokens.append(row)
    tokens = np.array(tokens)[:,1:-1].reshape(-1)    
    return (tokens, emb_lookup)

def unpad_prompt(end_token, prompt):
    return np.trim_zeros(prompt-end_token)+end_token

class CLIPRegionsBasePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP", )}}
    RETURN_TYPES = ("CLIPREGION",)
    FUNCTION = "init_prompt"

    CATEGORY = "conditioning/cutoff"

    def init_prompt(self, clip, text):
        tokens = clip.tokenizer.tokenize_with_weights(text)
        return ({
            "clip" : clip,
            "base_tokens" : tokens,
            "regions" : [],
            "targets" : [],
            "weights" : [],
        },)

def get_sublists(super_list, sub_list):
  positions = []
  for candidate_ind in (i for i,e in enumerate(super_list) if e==sub_list[0]):
    if super_list[candidate_ind:candidate_ind+len(sub_list)] == sub_list:
      positions.append(candidate_ind)
  return positions

class CLIPSetRegion:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip_regions": ("CLIPREGION", ),
                             "region_text": ("STRING", {"multiline": True}),
                             "target_text": ("STRING", {"multiline": False}), 
                             "weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05})}}
    RETURN_TYPES = ("CLIPREGION",)
    FUNCTION = "add_clip_region"

    CATEGORY = "conditioning/cutoff"

    def add_clip_region(self, clip_regions, region_text, target_text, weight):
        clip = clip_regions["clip"]
        region_outputs = []
        target_outputs = []

        #strip input strings
        region_text = region_text.strip()
        target_text = target_text.strip()

        prompt_tokens, emb_lookup = replace_embeddings(clip.tokenizer.end_token, clip_regions["base_tokens"])
        
        for rt in region_text.split('\n'):
            region_tokens = clip.tokenizer.tokenize_with_weights(rt)
            region_tokens, _ = replace_embeddings(clip.tokenizer.end_token, region_tokens, emb_lookup)
            region_tokens = unpad_prompt(clip.tokenizer.end_token, region_tokens).tolist()

            #calc region mask
            region_length = len(region_tokens)
            regions = get_sublists(list(prompt_tokens), region_tokens)

            region_mask = np.zeros(len(prompt_tokens))
            for r in regions:
                region_mask[r:r+region_length] = 1
            region_mask = region_mask.reshape(-1,clip.tokenizer.max_length-2)
            region_mask = np.pad(region_mask, pad_width=((0,0),(1,1)), mode='constant', constant_values=0)
            region_mask = region_mask.reshape(1, -1)
            region_outputs.append(region_mask)

            #calc target mask
            targets = []
            for target in target_text.split(" "):
                # deal with underscores
                target = re.sub(r"(?<!\\)_", " ", target)
                target = re.sub(r"\\_", "_", target)

                target_tokens = clip.tokenizer.tokenize_with_weights(target)
                target_tokens, _ = replace_embeddings(clip.tokenizer.end_token, target_tokens, emb_lookup)
                target_tokens = unpad_prompt(clip.tokenizer.end_token, target_tokens).tolist()

                targets.extend([(x, len(target_tokens)) for x in get_sublists(region_tokens, target_tokens)])
            targets = [(t_start + r, t_start + t_end + r) for r in regions for t_start, t_end in targets]

            targets_mask = np.zeros(len(prompt_tokens))
            for t_start, t_end in targets:
                targets_mask[t_start: t_end] = 1
            targets_mask = targets_mask.reshape(-1,clip.tokenizer.max_length-2)
            targets_mask = np.pad(targets_mask, pad_width=((0,0),(1,1)), mode='constant', constant_values=0)
            targets_mask = targets_mask.reshape(1,-1)
            target_outputs.append(targets_mask)

        #prepare output
        region_mask_list = clip_regions['regions'].copy()
        region_mask_list.extend(region_outputs)
        target_mask_list = clip_regions['targets'].copy()
        target_mask_list.extend(target_outputs)
        weight_list = clip_regions['weights'].copy()
        weight_list.extend([weight]*len(region_outputs))

        return ({
            "clip" : clip,
            "base_tokens" : clip_regions["base_tokens"],
            "regions" : region_mask_list,
            "targets" : target_mask_list,
            "weights" : weight_list,
        },)

def weighted_tokens_to_embedding(clip, tokens):
    if clip.layer_idx is not None:
        clip.cond_stage_model.clip_layer(clip.layer_idx)
    try:
        clip.patcher.patch_model()
        emb = clip.cond_stage_model.encode_token_weights(tokens)
        clip.patcher.unpatch_model()
    except Exception as e:
        clip.patcher.unpatch_model()
        raise e
    return emb

def create_masked_prompt(weighted_tokens, mask, mask_token):
    mask_ids = list(zip(*np.nonzero(mask)))  
    new_prompt = copy.deepcopy(weighted_tokens)
    for x,y in mask_ids:
        new_prompt[x][y] = (mask_token, new_prompt[x][y][1])
    return new_prompt

class CLIPRegionsToConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip_regions": ("CLIPREGION", ),
                             "mask_token": ("STRING", {"multiline": False, "default" : ""}),
                             "strict_mask": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "start_from_masked": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05})}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "finalize_clip_regions"

    CATEGORY = "conditioning/cutoff"

    def finalize_clip_regions(self, clip_regions, mask_token, strict_mask, start_from_masked):
        clip = clip_regions["clip"]
        base_weighted_tokens = clip_regions["base_tokens"]
        if mask_token == "":
            mask_token = clip.tokenizer.end_token
        else:
            mask_token = clip.tokenizer.tokenizer(mask_token)['input_ids'][1:-1]
            if len(mask_token) > 1:
                warnings.warn("mask_token does not map to a single token, using the first token instead")
            mask_token = mask_token[0]
            
        #calc global target mask
        global_target_mask = np.any(np.stack(clip_regions["targets"]), axis=0).astype(int)

        #calc global region mask
        global_region_mask = np.any(np.stack(clip_regions["regions"]), axis=0).astype(float)
        regions_sum = np.sum(np.stack(clip_regions["regions"]), axis=0)
        regions_normalized = np.divide(1, regions_sum, out=np.zeros_like(regions_sum), where=regions_sum!=0)

        #calc base embedding
        base_embedding_full = weighted_tokens_to_embedding(clip, base_weighted_tokens)
        base_embedding_masked = weighted_tokens_to_embedding(clip, create_masked_prompt(base_weighted_tokens, global_target_mask, mask_token))
        base_embedding_start = base_embedding_full * (1-start_from_masked) + base_embedding_masked * start_from_masked
        base_embedding_outer = base_embedding_full * (1-strict_mask) + base_embedding_masked * strict_mask

        region_embeddings = []
        for region, target, weight in zip (clip_regions["regions"],clip_regions["targets"],clip_regions["weights"]):
            region_masking = torch.tensor(regions_normalized * region * weight, dtype=base_embedding_full.dtype, device=base_embedding_full.device).unsqueeze(-1)

            region_emb = weighted_tokens_to_embedding(clip, create_masked_prompt(base_weighted_tokens, global_target_mask - target, mask_token))
            region_emb -= base_embedding_start
            region_emb *= region_masking

            region_embeddings.append(region_emb)
        region_embeddings = torch.stack(region_embeddings).sum(axis=0)

        embeddings_final_mask = torch.tensor(global_region_mask, dtype=base_embedding_full.dtype, device=base_embedding_full.device).unsqueeze(-1)
        embeddings_final = base_embedding_start * embeddings_final_mask + base_embedding_outer * (1 - embeddings_final_mask)
        embeddings_final += region_embeddings
        return ([[embeddings_final, {}]], )
    
NODE_CLASS_MAPPINGS = {
    "CLIPRegionsBasePrompt": CLIPRegionsBasePrompt,
    "CLIPSetRegion": CLIPSetRegion,
    "CLIPRegionsToConditioning": CLIPRegionsToConditioning,
}