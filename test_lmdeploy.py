#!/usr/bin/env python3

import torch
from lmdeploy import pipeline, GenerationConfig

def test_lmdeploy():
    print('Testing LMDeploy with RTX 5070 Ti...')
    print('PyTorch version:', torch.__version__)
    print('CUDA available:', torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
        print('GPU name:', torch.cuda.get_device_properties(0).name)
        print('GPU compute capability:', torch.cuda.get_device_capability(0))
    
    try:
        print('Initializing LMDeploy pipeline with a small model...')
        # Use a small model for testing with simpler configuration
        pipe = pipeline('microsoft/DialoGPT-small')
        
        print('Testing text generation...')
        response = pipe(['Hello, how are you today?'])
        
        print('LMDeploy test successful!')
        print('Response:', response)
        return True
        
    except Exception as e:
        print('Error during LMDeploy test:', str(e))
        print('Trying with a different model...')
        
        try:
            # Try with a different small model
            pipe = pipeline('gpt2')
            response = pipe(['Hello, how are you today?'])
            print('LMDeploy test successful with gpt2!')
            print('Response:', response)
            return True
        except Exception as e2:
            print('Error with gpt2:', str(e2))
            return False

if __name__ == '__main__':
    success = test_lmdeploy()
    if success:
        print('\n✅ LMDeploy setup is working correctly!')
    else:
        print('\n❌ LMDeploy setup has issues.') 