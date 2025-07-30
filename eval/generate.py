import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    Using float16 for better performance while maintaining reasonable quality.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0

    # Use float32 instead of float64 for better performance
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits, dtype=torch.float32)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    tokenizer,
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Optimized version of the generate function.
    """
    # Use mixed precision for faster computation
    with torch.autocast(device_type="cuda"):
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=prompt.device
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = x != mask_id

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        for num_block in tqdm(range(num_blocks), disable=(dist.get_rank() != 0)):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                # Handle classifier-free guidance more efficiently
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    # Get logits in a single forward pass
                    logits = model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits

                # Apply Gumbel noise for sampling
                logits_with_noise = add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Handle remasking strategy
                if remasking == "low_confidence":
                    # Use float32 instead of float64 for better performance
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand(x0.shape, device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                # Ensure we don't process tokens beyond the current block
                x0_p[:, end_idx:] = -np.inf

                # Update masked tokens
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, torch.tensor(-np.inf, device=x0.device))

                # Select tokens to transfer based on confidence
                for j in range(confidence.shape[0]):
                    num_tokens = num_transfer_tokens[j, i].item()
                    if num_tokens > 0:
                        _, select_indices = torch.topk(confidence[j], k=num_tokens)
                        x[j, select_indices] = x0[j, select_indices]
        return x
        


        
####################
# Avobe is prior code of d1, which does not allow to change "steps" for higher value than gen_length 
#####################

# import torch, random
# import numpy as np
# import torch.nn.functional as F

# from transformers import AutoTokenizer, AutoModel
# import math
# tokenizer = AutoTokenizer.from_pretrained('/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Instruct', trust_remote_code=True)


# def add_gumbel_noise(logits, temperature):
#     '''
#     The Gumbel max is a method for sampling categorical distributions.
#     According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
#     Thus, we use float64.
#     '''
#     if temperature == 0:
#         return logits
#     logits = logits.to(torch.float64)
#     noise = torch.rand_like(logits, dtype=torch.float64)
#     gumbel_noise = (- torch.log(noise)) ** temperature
#     return logits.exp() / gumbel_noise


# def standard_normal_cdf(z):
#     # CDF of the standard normal
#     return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


# def get_num_transfer_tokens(mask_index, steps):
#     '''
#     In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
#     Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
#     the expected number of tokens transitioned at each step should be consistent.

#     This function is designed to precompute the number of tokens that need to be transitioned at each step.
#     '''
#     mask_num = mask_index.sum(dim=1, keepdim=True)
#     base = mask_num // steps
#     remainder = mask_num % steps

#     ################### LLADA code
#     num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
#     for i in range(mask_num.size(0)):
#         num_transfer_tokens[i, :remainder[i]] += 1
#     return num_transfer_tokens

#     ################### d1 code
#     # num_transfer_tokens = base.expand(-1, steps).clone()
#     # if remainder.sum() > 0:
#     #     indices = torch.arange(steps, device=mask_index.device)
#     #     mask = indices.unsqueeze(0) < remainder
#     #     num_transfer_tokens[mask] += 1
#     # return num_transfer_tokens.to(torch.int64)


# @ torch.no_grad()
# @torch.no_grad()
# def generate(model, prompt, tokenizer, steps=128, gen_length=128, block_length=128, temperature=0.,
#              cfg_scale=0., remasking='low_confidence', mask_id=126336):
#     """
#     Batched version of generate(). Supports arbitrary batch sizes.

#     Args:
#         model: Mask predictor.
#         prompt: Tensor of shape (B, L).
#         steps: Sampling steps.
#         gen_length: Generated answer length.
#         block_length: Block length.
#         temperature: Sampling temperature.
#         cfg_scale: Classifier-free guidance scale.
#         remasking: Remasking strategy.
#         mask_id: [MASK] token ID.
#     """
#     B, L = prompt.shape  # batch size, prompt length
#     x = torch.full((B, L + gen_length), mask_id, dtype=torch.long).to(model.device)
#     x[:, :L] = prompt.clone()

#     prompt_index = (x != mask_id)

#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length
#     assert steps % num_blocks == 0
#     steps = steps // num_blocks

#     for num_block in range(num_blocks):
#         block_start = L + num_block * block_length
#         block_end = L + (num_block + 1) * block_length

#         block_mask_index = (x[:, block_start:block_end] == mask_id)
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

#         for i in range(steps):
#             mask_index = (x == mask_id)

#             if cfg_scale > 0.:
#                 un_x = x.clone()
#                 un_x[prompt_index] = mask_id
#                 x_ = torch.cat([x, un_x], dim=0)
#                 logits = model(x_).logits
#                 logits, un_logits = torch.chunk(logits, 2, dim=0)
#                 logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#             else:
#                 logits = model(x).logits

#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1)

#             if remasking == 'low_confidence':
#                 p = F.softmax(logits, dim=-1)
#                 x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
#             elif remasking == 'low_confidence_w_rescoring':
#                 p = F.softmax(logits, dim=-1)
#                 x0_p_all = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
#                 x0_p = x0_p_all
#             elif remasking == 'bernoulli_scaled':
#                 p_dist = F.softmax(logits, dim=-1)
#                 conf = torch.gather(p_dist, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
#                 scale = (steps - i) / float(steps)
#                 prob = (1.0 - conf) * scale
#                 bern_mask = torch.bernoulli(prob).bool()
#                 block_mask = torch.zeros_like(bern_mask)
#                 block_mask[:, block_start:block_end] = 1
#                 bern_mask &= block_mask.bool()
#                 x = torch.where(bern_mask, x0, x)
#                 continue
#             elif remasking == 'random':
#                 x0_p = torch.rand_like(x0, dtype=torch.float)
#             elif remasking == 'gaussian':
#                 x0_p = standard_normal_cdf(torch.randn_like(x0, dtype=torch.float))
#             elif remasking == 'beta':
#                 alpha, beta_param = 0.9, 3.5
#                 beta_dist = torch.distributions.Beta(alpha, beta_param)
#                 x0_p = beta_dist.sample((B, x0.shape[1])).to(x0.device)
#             elif remasking == 'chi_squared':
#                 df = 6
#                 chi_dist = torch.distributions.Chi2(df=df)
#                 x0_p = chi_dist.sample((B, x0.shape[1])).to(x0.device)
#                 x0_p = 1. - x0_p / (x0_p.max(dim=-1, keepdim=True).values + 1e-6)
#             else:
#                 raise NotImplementedError(remasking)

#             # Mask everything beyond current block
#             x0_p[:, block_end:] = -np.inf
#             x0 = torch.where(mask_index, x0, x)

#             if remasking == 'low_confidence_w_rescoring':
#                 min_conf_threshold = 0.15
#                 confidence_mask = (x0_p < min_conf_threshold)
#                 confidence = torch.where(confidence_mask, x0_p, torch.tensor(-np.inf, device=x0.device))
#             else:
#                 confidence = torch.where(mask_index, x0_p, -np.inf)

#             transfer_index = torch.zeros_like(x0, dtype=torch.bool)
#             for j in range(B):
#                 valid_token_count = (confidence[j] != -np.inf).sum().item()
#                 k = min(num_transfer_tokens[j, i].item(), valid_token_count)
#                 if k > 0:
#                     _, select_index = torch.topk(confidence[j], k=k)
#                     transfer_index[j, select_index] = True
#                 else:
#                     fallback_mask = mask_index[j].nonzero(as_tuple=False).squeeze(-1)
#                     if fallback_mask.numel() > 0:
#                         rand_select = fallback_mask[torch.randint(0, fallback_mask.numel(), (1,))]
#                         transfer_index[j, rand_select] = True

#             x[transfer_index] = x0[transfer_index]

#             ###############
#             # decoded = tokenizer.batch_decode(x[:, L:], skip_special_tokens=False)
#             # for b in range(B):
#             #     masked = decoded[b].replace("<|mdm_mask|>", "🟨")
#             #     print(f"[{b}] Step {i}/{steps} Block {num_block}: {masked}")
#             ###############
#     return x



def main():

    def set_seed(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(42)

    device = 'cuda'

    # model = AutoModel.from_pretrained('/home/work/jihoon_wombat_storage/JIHOON/d1/diffu-grpo/merged_models/checkpoint-44800_merged', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    # tokenizer = AutoTokenizer.from_pretrained('/home/work/jihoon_wombat_storage/JIHOON/d1/diffu-grpo/merged_models/checkpoint-44800_merged', trust_remote_code=True)

    model = AutoModel.from_pretrained('/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Base', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Base', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    prompt = "John runs 60 miles a week. He runs 3 days a week.  He runs 3 hours the first day and half as much the other two days he runs.  How fast does he run?"
    prompt = 'What is the answer of 333 * 333?'
    
    
    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, tokenizer, steps=32, gen_length=32, block_length=8, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
