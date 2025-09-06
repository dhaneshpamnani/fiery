# FIERY Inference Testing Script

This script (`test_first.py`) runs inference on the FIERY model using the validation set and saves all outputs with performance timing.

## Usage

```bash
python test_first.py --checkpoint <path_to_checkpoint> --dataroot <path_to_dataset> --version <mini|trainval>
```

### Examples

For NuScenes mini dataset:
```bash
python test_first.py --checkpoint ./fiery.ckpt --dataroot ./nuscenes --version mini
```

For full NuScenes dataset:
```bash
python test_first.py --checkpoint ./fiery.ckpt --dataroot ./nuscenes --version trainval
```

## Output Structure

The script creates a `results_first/` folder with the following structure:

```
results_first/
├── mini/ (or trainval/)
│   ├── scene_0001/
│   │   ├── sample_000/
│   │   │   ├── segmentation.npy
│   │   │   ├── instance_center.npy
│   │   │   ├── instance_offset.npy
│   │   │   ├── instance_flow.npy
│   │   │   ├── present_mu.npy
│   │   │   ├── present_log_sigma.npy
│   │   │   ├── future_mu.npy
│   │   │   └── future_log_sigma.npy
│   │   └── sample_001/
│   ├── scene_0002/
│   │   ├── sample_000/
│   │   └── sample_001/
│   ├── visualizations/
│   │   ├── scene_0001_sample_000.png
│   │   ├── scene_0001_sample_001.png
│   │   └── ...
│   └── timing_log.json
```

## Features

- **Full Dataset Processing**: Uses the complete mini dataset (not limited to 10 samples)
- **Performance Timing**: Measures inference time per sample and overall performance
- **Complete Output Saving**: Saves all model outputs (segmentation, instance, flow, probabilistic outputs)
- **Visualization**: Creates visualization images showing camera views + BEV predictions
- **Detailed Logging**: JSON file with comprehensive timing and performance metrics

## Performance Metrics

The script tracks:
- Average inference time per sample
- FPS (Frames Per Second)
- Min/Max/Std deviation of inference times
- Per-scene breakdown of timing
- Total processing time

## Requirements

- PyTorch
- OpenCV
- Matplotlib
- PIL
- NumPy
- tqdm
- All FIERY dependencies

## Notes

- The script uses mean prediction (noise=0) for consistent results
- All outputs are saved as NumPy arrays for easy loading
- Visualizations are saved as PNG images
- Timing data is saved as JSON for easy analysis
