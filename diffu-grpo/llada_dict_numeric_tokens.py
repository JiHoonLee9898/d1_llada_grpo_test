import torch
import numpy as np
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
import math

device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(
    '/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Base',
    trust_remote_code=True
)

# numeric_token_indices = []
# vocab = tokenizer.get_vocab()  # {token_string: token_id}
# numeric_tokens = [(token, idx) for token, idx in vocab.items() if any(c.isdigit() for c in token)]
# numeric_tokens = sorted(numeric_tokens, key=lambda x: len(x[0]))

# print(f"✅ 숫자가 포함된 토큰 수: {len(numeric_tokens)}\n")
# for token, idx in numeric_tokens[:100]:  # 너무 많을 수 있으니 100개만 출력
#     print(f"{token} --> {idx} | len: {len(token)}")
#     numeric_token_indices.append(idx)

# print(numeric_token_indices)


# # 임의의 입력 문자열
# input_text = "Lily can run 12 kilometers per hour for 4 hours."

# # 토크나이징
# tokens = tokenizer.tokenize(input_text)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)

# # 출력
# for token, token_id in zip(tokens, token_ids):
#     print(f"Token: {token:<15} ID: {token_id}")