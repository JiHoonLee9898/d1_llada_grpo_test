from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil
import os


base_model_path = "/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Base"
lora_adapter_path = "/home/work/jihoon_wombat_storage/JIHOON/d1/diffu-grpo/checkpoints/gsm8k_base_bs12/checkpoint-44800"
merged_model_path = "/home/work/jihoon_wombat_storage/JIHOON/d1/diffu-grpo/merged_models/checkpoint-44800_merged"
modeling_llada_py_path = "/home/work/jihoon_wombat_storage/MODELS/LLaDA-8B-Base/modeling_llada.py"


base_model = AutoModelForCausalLM.from_pretrained(base_model_path,device_map="auto")
lora_model = PeftModel.from_pretrained(base_model,lora_adapter_path)

merged_model = lora_model.merge_and_unload()

merged_model.save_pretrained(merged_model_path)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)

#####

source_file = modeling_llada_py_path
destination_file = os.path.join(merged_model_path, "modeling_llada.py")
shutil.copy2(source_file, destination_file)
print(f"Copied to: {destination_file}")
