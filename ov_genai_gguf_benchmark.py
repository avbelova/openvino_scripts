import openvino_genai as ov_genai
import openvino as ov
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time
import argparse
from huggingface_hub import hf_hub_download

prompt = 'Why is the Sun yellow?'
#device = "CPU"

parser = argparse.ArgumentParser()
parser.add_argument("model_id")
parser.add_argument("model_file")
parser.add_argument("device")
parser.add_argument("iter_num")
args = parser.parse_args()

model_id = args.model_id
model_file = args.model_file
device = args.device
iter_num = int(args.iter_num)

file_path = hf_hub_download(repo_id = model_id, filename = model_file, cache_dir="./models")

hf_tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file = model_file)

ov_generation_config = ov_genai.GenerationConfig()
ov_generation_config.max_new_tokens = 30
ov_generation_config.apply_chat_template = False
ov_generation_config.set_eos_token_id(hf_tokenizer.eos_token_id)

tokenization_start = time.perf_counter()
inputs = hf_tokenizer(prompt, return_tensors="pt")
tokenization_end = time.perf_counter()
tokenization_time = [(tokenization_end - tokenization_start) * 1000]
input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']

pipe = ov_genai.LLMPipeline(file_path, device)
ttfts=[]
ttsts=[]
for i in range(iter_num):
    encoded_result  = pipe.generate(ov.Tensor(input_ids.numpy()), generation_config=ov_generation_config)
    result_string = hf_tokenizer.batch_decode([encoded_result.tokens[0]], skip_special_tokens=True)[0]
    print(result_string)
    perf_metrics = encoded_result.perf_metrics
    ttft = perf_metrics.get_ttft().mean
    ttfts.append(ttft)
    print("TTFT: ", ttft)
    other_tokens_duration = np.mean((
        np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
        - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1])
    ))
    print("Other tokens latency", other_tokens_duration)
    ttsts.append(other_tokens_duration)

print("Average TTFT: ", round(sum(ttfts[1:]) / (len(ttfts) - 1)), "ms")
print("Average other tokens duration: ", round(sum(ttsts[1:]) / (len(ttsts) - 1)), "ms")