# Multi-Model LM Studio Setup Guide

This guide explains how to set up multiple models in LM Studio to run concurrent Q&A generation, potentially halving (or more) your processing time.

## Prerequisites

- LM Studio with multiple models loaded
- Sufficient GPU memory to run multiple models simultaneously
- Models loaded and available in the same LM Studio instance

## Setup Instructions

### Option 1: Multiple Models in Single LM Studio Instance (Recommended)

1. **Start LM Studio Instance**
   - Open LM Studio
   - Load your first model (e.g., `qwen3-4b-mlx`)
   - Load your second model (it will get identifier like `qwen3-4b-mlx:2`)
   - Go to Local Server tab
   - Set port to `1234`
   - Start the server

2. **Verify Models are Available**
   - In LM Studio, check that both models show up in the model list
   - Note the exact model identifiers (including any `:2`, `:3` suffixes)

3. **Update Configuration in Script**
   ```python
   MODEL_CONFIGS = [
       {
           "name": "qwen3-4b-mlx",
           "base_url": "http://192.168.1.72:1234/v1",
           "api_key": "lm-studio",
           "port": 1234
       },
       {
           "name": "qwen3-4b-mlx:2",  # Note the :2 suffix
           "base_url": "http://192.168.1.72:1234/v1",  # Same URL
           "api_key": "lm-studio",
           "port": 1234
       }
   ]
   ```

### Option 2: Multiple LM Studio Instances on Different Ports

1. **Start First LM Studio Instance (Port 1234)**
   - Open LM Studio
   - Load your first model (e.g., `qwen3-4b-mlx`)
   - Go to Local Server tab
   - Set port to `1234`
   - Start the server

2. **Start Second LM Studio Instance (Port 1235)**
   - Open another LM Studio instance
   - Load your second model (e.g., `devstral-small-2505-mlx`)
   - Go to Local Server tab
   - Set port to `1235`
   - Start the server

3. **Update Configuration in Script**
   ```python
   MODEL_CONFIGS = [
       {
           "name": "qwen3-4b-mlx",
           "base_url": "http://192.168.1.72:1234/v1",
           "api_key": "lm-studio",
           "port": 1234
       },
       {
           "name": "devstral-small-2505-mlx", 
           "base_url": "http://192.168.1.72:1235/v1",
           "api_key": "lm-studio",
           "port": 1235
       }
   ]
   ```

### Configuration Options

- **USE_CONCURRENT_PROCESSING**: Set to `True` to enable concurrent processing
- **MAX_WORKERS**: Automatically set to the number of available models
- **BATCH_SIZE**: Adjust based on your GPU memory

## Performance Expectations

- **Single Model**: Baseline processing speed
- **Two Models Concurrent**: ~2x speedup (theoretical)
- **Three+ Models**: Further speedup limited by GPU memory and bandwidth

## GPU Memory Considerations

- **RTX 5070 Ti (16GB)**: Can typically handle 2-3 small models (3-4B parameters)
- **RTX 4090 (24GB)**: Can handle 2-4 models depending on size
- **Monitor GPU memory usage** to avoid OOM errors

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Check exact model names in LM Studio (including `:2`, `:3` suffixes)
   - Update MODEL_CONFIGS with exact model identifiers
   - Ensure models are fully loaded

2. **GPU Memory Exhausted**
   - Reduce number of concurrent models
   - Use smaller models
   - Reduce BATCH_SIZE

3. **Connection Errors**
   - Verify LM Studio instance is running
   - Check firewall settings
   - Ensure correct IP addresses and ports

### Verification

Run the script and look for:
```
Successfully configured X models!
Concurrent processing: Enabled
Available models: ['qwen3-4b-mlx', 'qwen3-4b-mlx:2']
Max concurrent workers: 2
```

## Performance Monitoring

The script will show:
- Processing time per batch
- Q&A pairs generated per model
- Overall throughput improvement
- Model distribution statistics

## Example Output

```
=== Performance Notes ===
Framework: LM Studio API (OpenAI-compatible)
Concurrent processing: Enabled
Models used: 2
Expected speedup: ~2x faster than single model

Q&A pairs by model:
  qwen3-4b-mlx: 1,250
  qwen3-4b-mlx:2: 1,180
```

## Tips for Optimal Performance

1. **Load Balance**: Use models of similar speed for best load balancing
2. **Memory Management**: Monitor GPU memory usage during processing
3. **Network**: Use localhost connections for best performance
4. **Model Selection**: Choose complementary models for diverse Q&A generation
5. **Batch Size**: Experiment with different batch sizes for optimal throughput
6. **Model Naming**: Pay attention to exact model identifiers in LM Studio 