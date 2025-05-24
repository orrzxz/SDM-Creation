#!/usr/bin/env python3

from vllm import LLM, SamplingParams
import torch

def test_vllm():
    print('Testing vLLM with hardware setup...')
    print('CUDA available:', torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
        print('GPU name:', torch.cuda.get_device_properties(0).name)
    
    # Test with a very small model first
    try:
        print('Initializing vLLM with microsoft/DialoGPT-small...')
        llm = LLM(
            model='microsoft/DialoGPT-small', 
            gpu_memory_utilization=0.3, 
            max_model_len=512,
            trust_remote_code=True
        )
        
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        outputs = llm.generate(['Hello, how are you?'], sampling_params)
        
        print('vLLM test successful!')
        print('Response:', outputs[0].outputs[0].text[:100])
        return True
        
    except Exception as e:
        print('Error during vLLM test:', str(e))
        return False

if __name__ == '__main__':
    success = test_vllm()
    if success:
        print('\n✅ vLLM setup is working correctly!')
    else:
        print('\n❌ vLLM setup has issues.') 