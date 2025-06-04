import openvino_genai as ov_genai
import openvino as ov
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import time

prompt = 'Why is the Sun yellow?'
device = "CPU"

hf_tokenizer = AutoTokenizer.from_pretrained('bartowski/Meta-Llama-3.1-8B-Instruct-GGUF', gguf_file='Meta-Llama-3.1-8B-Instruct-f32.gguf')

ov_generation_config = ov_genai.GenerationConfig()
ov_generation_config.max_new_tokens = 30
ov_generation_config.apply_chat_template = False
ov_generation_config.set_eos_token_id(hf_tokenizer.eos_token_id)

tokenization_start = time.perf_counter()
inputs = hf_tokenizer(prompt, return_tensors="pt")
tokenization_end = time.perf_counter()
tokenization_time = [(tokenization_end - tokenization_start) * 1000]
input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']

pipe = ov_genai.LLMPipeline("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", device)
encoded_result  = pipe.generate(ov.Tensor(input_ids.numpy()), generation_config=ov_generation_config)
res_string_input_2 = hf_tokenizer.batch_decode([encoded_result.tokens[0]], skip_special_tokens=True)[0]
print(res_string_input_2)
perf_metrics = encoded_result.perf_metrics
print("TTFT: ", perf_metrics.get_ttft().mean)
second_tokens_durations = (
        np.array(perf_metrics.raw_metrics.m_new_token_times[1:])
        - np.array(perf_metrics.raw_metrics.m_new_token_times[:-1])
    ).tolist()
print("Second Token ", second_tokens_durations)
)
