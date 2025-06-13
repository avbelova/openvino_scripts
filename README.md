# openvino_scripts
This repo is a collection of usefull OpenVINO-related scripts for dufferent purposes. Below you can find an information about each of them.

## ov_genai_gguf_benchmark.py 
It helps to benchmark .gguf models with OpenVINO.GenAI. 
Usage: 
```
python ov_genai_gguf_benchmark.py <HF model id> <model file> <device (CPU, GPU or NPU)> <number of iterations>
```
Example:
```
python ov_genai_gguf_benchmark.py unsloth/Llama-3.2-1B-Instruct-GGUF Llama-3.2-1B-Instruct-F16.gguf CPU 3
```
