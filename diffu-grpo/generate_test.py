import torch, random
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import math
tokenizer = AutoTokenizer.from_pretrained('/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Instruct', trust_remote_code=True)


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def standard_normal_cdf(z):
    # CDF of the standard normal
    return 0.5 * (1 + torch.erf(z / math.sqrt(2)))


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    ################### LLADA code
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens

    ################### d1 code
    # num_transfer_tokens = base.expand(-1, steps).clone()
    # if remainder.sum() > 0:
    #     indices = torch.arange(steps, device=mask_index.device)
    #     mask = indices.unsqueeze(0) < remainder
    #     num_transfer_tokens[mask] += 1
    # return num_transfer_tokens.to(torch.int64)


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'low_confidence_w_rescoring':
                p = F.softmax(logits, dim=-1)
                x0_p_all = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                x0_p = x0_p_all
            elif remasking == 'bernoulli_scaled':
                p_dist = F.softmax(logits, dim=-1)
                conf = torch.gather(p_dist, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (b, l)

                scale = (steps - i) / float(steps)  # e.g. 1.0, 0.9, ..., 1/steps
                prob = (1.0 - conf) * scale
                
                bern_mask = torch.bernoulli(prob).bool()
                block_start = prompt.shape[1] + num_block * block_length
                block_end = prompt.shape[1] + (num_block + 1) * block_length

                block_mask = torch.zeros_like(bern_mask)
                block_mask[:, block_start:block_end] = 1
                bern_mask = bern_mask & block_mask.bool()
                x = torch.where(bern_mask, x0, x)
                continue
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'gaussian':
                # Draw from standard normal, then map to [0, 1] via CDF
                x0_p = torch.randn((x0.shape[0], x0.shape[1]), device=x0.device)
                x0_p = standard_normal_cdf(x0_p)  # Shape (b, l), in (0,1)
            elif remasking == 'beta':
                # Beta distribution in [0,1]
                alpha, beta_param = 0.9, 3.5
                beta_dist = torch.distributions.Beta(alpha, beta_param)
                x0_p = beta_dist.sample((x0.shape[0], x0.shape[1])).to(x0.device)  # Shape (b, l)
            elif remasking == 'chi_squared':
                # Chi-squared is [0, ∞), so we need to normalize to [0,1]
                df = 6
                chi_dist = torch.distributions.Chi2(df=df)
                x0_p = chi_dist.sample((x0.shape[0], x0.shape[1])).to(x0.device)  # Shape (b, l)
                # Normalize by max value in each row to [0,1], then invert for "low is high confidence"
                x0_p = x0_p / (x0_p.max(dim=-1, keepdim=True).values + 1e-6)
                x0_p = 1. - x0_p  # So that "large value" = "low confidence" (like in the others)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            if remasking == 'low_confidence_w_rescoring':
                min_conf_threshold = 0.15 # add threshold to prevent stalling
                confidence_mask = (x0_p < min_conf_threshold)
                confidence = torch.where(confidence_mask, x0_p, torch.tensor(-np.inf, device=x0_p.device))
            else:
                confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                valid_token_count = (confidence[j] != -np.inf).sum().item()
                k = min(num_transfer_tokens[j, i].item(), valid_token_count)
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True
                else:
                    fallback_mask = mask_index[j].nonzero(as_tuple=False).squeeze(-1)
                    if fallback_mask.numel() > 0:
                        rand_select = fallback_mask[torch.randint(0, fallback_mask.numel(), (1,))]
                        transfer_index[j, rand_select] = True

            x[transfer_index] = x0[transfer_index]
            input_ids = prompt 
            decoded = tokenizer.batch_decode(x[:, input_ids.shape[1]:], skip_special_tokens=False)[0]
            masked = decoded.replace("<|mdm_mask|>", "\U0001F7E8") # 🟨
            
            
            print('-'*50)
            print(f"num_block: {num_block} | decoding step: {i} | remasked: {masked}")
            print('-'*50)



    # log_probs = F.log_softmax(logits, dim=-1)
    # gen_log_probs= log_probs[:, prompt.shape[1]:, :]
    # gen_ids = x[:, prompt.shape[1]:].unsqueeze(-1)

    # ll_per_token = gen_log_probs.gather(dim=-1, index=gen_ids)
    # ll_per_token = ll_per_token.squeeze(-1)

    # total_ll = ll_per_token.sum(dim=1) 
    return x #, total_ll


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

    model = AutoModel.from_pretrained('/home/work/jihoon_wombat_storage/JIHOON/d1_jihoon/diffu-grpo/merged_models/numeric_exclude_merged_epoch1', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/home/work/jihoon_wombat_storage/JIHOON/d1_jihoon/diffu-grpo/merged_models/numeric_exclude_merged_epoch1', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    prompt = "John runs 60 miles a week. He runs 3 days a week.  He runs 3 hours the first day and half as much the other two days he runs.  How fast does he run?"
    prompt = 'What is the answer of 333 * 333?'
    
    
    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(model, input_ids, steps=32, gen_length=32, block_length=8, temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()
