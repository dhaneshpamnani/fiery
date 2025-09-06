import os
import time
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import Image
from tqdm import tqdm

from fiery.data import prepare_dataloaders
from fiery.trainer import TrainingModule
from fiery.utils.network import preprocess_batch, NormalizeInverse
from fiery.utils.instance import predict_instance_segmentation_and_trajectories, convert_instance_mask_to_center_and_offset_label
from fiery.utils.visualisation import plot_instance_map, generate_instance_colours, make_contour, convert_figure_numpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl


def create_results_folder_structure(version):
    """Create the folder structure for storing results"""
    base_dir = Path("results_first")
    version_dir = base_dir / version
    
    # Create main directories
    version_dir.mkdir(parents=True, exist_ok=True)
    
    return base_dir, version_dir


def save_model_outputs(output, sample_token, scene_name, sample_number, version_dir):
    """Save model outputs for a single sample"""
    scene_dir = version_dir / scene_name
    sample_dir = scene_dir / f"sample_{sample_number:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Save different output components
    outputs_to_save = {}
    
    # Always save these core outputs
    if 'segmentation' in output and output['segmentation'] is not None:
        outputs_to_save['segmentation'] = output['segmentation'].detach().cpu().numpy()
    
    if 'instance_center' in output and output['instance_center'] is not None:
        outputs_to_save['instance_center'] = output['instance_center'].detach().cpu().numpy()
    
    if 'instance_offset' in output and output['instance_offset'] is not None:
        outputs_to_save['instance_offset'] = output['instance_offset'].detach().cpu().numpy()
    
    if 'instance_flow' in output and output['instance_flow'] is not None:
        outputs_to_save['instance_flow'] = output['instance_flow'].detach().cpu().numpy()
    
    if 'present_mu' in output and output['present_mu'] is not None:
        outputs_to_save['present_mu'] = output['present_mu'].detach().cpu().numpy()
        outputs_to_save['present_log_sigma'] = output['present_log_sigma'].detach().cpu().numpy()
    
    if 'future_mu' in output and output['future_mu'] is not None:
        outputs_to_save['future_mu'] = output['future_mu'].detach().cpu().numpy()
        outputs_to_save['future_log_sigma'] = output['future_log_sigma'].detach().cpu().numpy()
    
    # Save each output as numpy file
    for key, value in outputs_to_save.items():
        np.save(sample_dir / f"{key}.npy", value)
    
    return sample_dir


def create_visualization(image, output, cfg, sample_token, scene_name, sample_number, version_dir):
    """Create and save visualization for a single sample"""
    # Process predictions
    consistent_instance_seg, matched_centers = predict_instance_segmentation_and_trajectories(
        output, compute_matched_centers=True
    )

    # Plot future trajectories
    unique_ids = torch.unique(consistent_instance_seg[0, 0]).cpu().long().numpy()[1:]
    instance_map = dict(zip(unique_ids, unique_ids))
    instance_colours = generate_instance_colours(instance_map)
    vis_image = plot_instance_map(consistent_instance_seg[0, 0].cpu().numpy(), instance_map)
    
    # Create trajectory overlay
    trajectory_img = np.zeros(vis_image.shape, dtype=np.uint8)
    for instance_id in unique_ids:
        if instance_id in matched_centers:
            path = matched_centers[instance_id]
            for t in range(len(path) - 1):
                color = instance_colours[instance_id].tolist()
                cv2.line(trajectory_img, tuple(map(int, path[t])), tuple(map(int, path[t + 1])), color, 4)

    # Overlay arrows
    temp_img = cv2.addWeighted(vis_image, 0.7, trajectory_img, 0.3, 1.0)
    mask = ~np.all(trajectory_img == 0, axis=2)
    vis_image[mask] = temp_img[mask]

    # Plot present RGB frames and predictions
    val_w = 2.99
    cameras = cfg.IMAGE.NAMES
    image_ratio = cfg.IMAGE.FINAL_DIM[0] / cfg.IMAGE.FINAL_DIM[1]
    val_h = val_w * image_ratio
    fig = plt.figure(figsize=(4 * val_w, 2 * val_h))
    width_ratios = (val_w, val_w, val_w, val_w)
    gs = mpl.gridspec.GridSpec(2, 4, width_ratios=width_ratios)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    denormalise_img = torchvision.transforms.Compose(
        (NormalizeInverse(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         torchvision.transforms.ToPILImage(),)
    )
    
    for imgi, img in enumerate(image[0, -1]):
        ax = plt.subplot(gs[imgi // 3, imgi % 3])
        showimg = denormalise_img(img.cpu())
        if imgi > 2:
            showimg = showimg.transpose(Image.FLIP_LEFT_RIGHT)

        plt.annotate(cameras[imgi].replace('_', ' ').replace('CAM ', ''), (0.01, 0.87), c='white',
                     xycoords='axes fraction', fontsize=14)
        plt.imshow(showimg)
        plt.axis('off')

    ax = plt.subplot(gs[:, 3])
    plt.imshow(make_contour(vis_image[::-1, ::-1]))
    
    # Mark ego position (center of BEV map)
    bev_height, bev_width = vis_image.shape[:2]
    ego_x, ego_y = bev_width // 2, bev_height // 2
    
    # Draw a car-like rectangle to mark ego position
    # Car dimensions: 6x3 pixels (length x width)
    car_length, car_width = 4, 8
    car_x1 = ego_x - car_length // 2
    car_y1 = ego_y - car_width // 2
    car_x2 = ego_x + car_length // 2
    car_y2 = ego_y + car_width // 2
    
    # Draw red filled rectangle (car body)
    car_rect = plt.Rectangle((car_x1, car_y1), car_length, car_width, 
                           color='red', fill=True, alpha=0.8)
    ax.add_patch(car_rect)
    
    # Draw black outline rectangle
    car_outline = plt.Rectangle((car_x1, car_y1), car_length, car_width, 
                              color='black', fill=False, linewidth=1)
    ax.add_patch(car_outline)
    
    plt.axis('off')

    plt.draw()
    figure_numpy = convert_figure_numpy(fig)
    plt.close()
    
    # Save visualization
    scene_dir = version_dir / scene_name
    vis_path = scene_dir / f"sample_{sample_number:03d}.png"
    Image.fromarray(figure_numpy).save(vis_path)
    
    return vis_path


def run_inference_on_validation_set(checkpoint_path, dataroot, version, save_results=False):
    """Main function to run inference on validation set"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Using dataroot: {dataroot}")
    print(f"Dataset version: {version}")
    print(f"Save results: {save_results}")
    
    # Load model
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainer.to(device)
    model = trainer.model

    # Update config
    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1
    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    # Create results folder structure only if saving is enabled
    base_dir, version_dir = None, None
    if save_results:
        base_dir, version_dir = create_results_folder_structure(version)
        print(f"Results will be saved to: {base_dir}")
    else:
        print("Results saving disabled - no files will be saved to disk")

    # Prepare data loaders (modify to remove 10-sample limitation for mini)
    _, valloader = prepare_dataloaders(cfg)
    
    # Remove the 10-sample limitation for mini dataset
    if version == 'mini':
        # Get the validation dataset and remove the limitation
        val_dataset = valloader.dataset
        # Reset indices to use full mini dataset
        val_dataset.indices = val_dataset.indices  # This will use all available mini samples
        print(f"Using full mini dataset with {len(val_dataset)} samples")

    # Get scene information for proper folder organization
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
    
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=False)
    splits = create_splits_scenes()
    val_scenes = splits[f'{version}_val']
    
    print(f"Found {len(val_scenes)} validation scenes: {val_scenes}")
    
    # Create mapping from sample tokens to scene names
    sample_to_scene = {}
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        scene_name = nusc.get('scene', scene_token)['name']
        if scene_name in val_scenes:
            sample_to_scene[sample['token']] = scene_name

    # Timing and performance tracking
    timing_data = {
        'total_samples': 0,
        'total_time': 0,
        'inference_times': [],
        'scene_times': {},
        'samples_per_scene': {}
    }

    print(f"Starting sequential inference on {len(valloader)} batches...")
    
    # Track samples per scene for sequential numbering
    scene_sample_counts = {}
    
    # Process each batch
    for batch_idx, batch in enumerate(tqdm(valloader, desc="Processing validation set")):
        batch_start_time = time.time()
        
        # Preprocess batch
        preprocess_batch(batch, device)
        
        # Get batch data
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']
        sample_tokens = batch['sample_token']

        batch_size = image.shape[0]
        
        # Process each sample in the batch
        for sample_idx in range(batch_size):
            sample_start_time = time.time()
            
            # Get single sample data
            single_image = image[sample_idx:sample_idx+1]
            single_intrinsics = intrinsics[sample_idx:sample_idx+1]
            single_extrinsics = extrinsics[sample_idx:sample_idx+1]
            single_future_egomotion = future_egomotion[sample_idx:sample_idx+1]
            sample_token = sample_tokens[sample_idx]
            
            # Get scene name from sample token
            scene_name = sample_to_scene.get(sample_token[0] if isinstance(sample_token, list) else sample_token, "unknown")
            
            # Initialize scene sample count if needed
            if scene_name not in scene_sample_counts:
                scene_sample_counts[scene_name] = 0
            scene_sample_counts[scene_name] += 1
            sample_number = scene_sample_counts[scene_name]
            
            # Run inference
            with torch.no_grad():
                # Use mean prediction (noise=0) for consistent results
                noise = torch.zeros((1, 1, model.latent_dim), device=device)
                output = model(single_image, single_intrinsics, single_extrinsics, 
                             single_future_egomotion, noise=noise)
                
                # Debug: Print available outputs (only for first sample)
                if timing_data['total_samples'] == 0:
                    print(f"Available model outputs: {list(output.keys())}")
                    for key, value in output.items():
                        if value is not None:
                            print(f"  {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else 'no shape'}")
                        else:
                            print(f"  {key}: None")

            # Measure inference time
            inference_time = time.time() - sample_start_time
            
            # Save model outputs and visualization only if saving is enabled
            sample_dir = None
            vis_path = None
            if save_results and version_dir is not None:
                sample_dir = save_model_outputs(output, sample_token, scene_name, sample_number, version_dir)
                vis_path = create_visualization(single_image, output, cfg, sample_token, scene_name, sample_number, version_dir)
            
            # Update timing data
            timing_data['total_samples'] += 1
            timing_data['total_time'] += inference_time
            timing_data['inference_times'].append(inference_time)
            
            # Track per-scene timing
            if scene_name not in timing_data['scene_times']:
                timing_data['scene_times'][scene_name] = 0
                timing_data['samples_per_scene'][scene_name] = 0
            timing_data['scene_times'][scene_name] += inference_time
            timing_data['samples_per_scene'][scene_name] += 1
            
            # Display timing information for each forward pass
            save_info = f" - Saved to: {sample_dir}" if sample_dir else " - No saving"
            print(f"Processed {scene_name} sample {sample_number:03d}: {sample_token} - "
                  f"Inference time: {inference_time:.4f}s{save_info}")

        batch_time = time.time() - batch_start_time
        print(f"Completed batch {batch_idx + 1}/{len(valloader)} in {batch_time:.2f}s")

    # Calculate performance metrics
    avg_inference_time = np.mean(timing_data['inference_times'])
    fps = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Save timing log
    timing_log = {
        'dataset_version': version,
        'total_samples': timing_data['total_samples'],
        'total_time_seconds': timing_data['total_time'],
        'average_inference_time_seconds': avg_inference_time,
        'fps': fps,
        'min_inference_time': np.min(timing_data['inference_times']),
        'max_inference_time': np.max(timing_data['inference_times']),
        'std_inference_time': np.std(timing_data['inference_times']),
        'scene_breakdown': {
            scene_name: {
                'samples': timing_data['samples_per_scene'][scene_name],
                'total_time': timing_data['scene_times'][scene_name],
                'avg_time_per_sample': timing_data['scene_times'][scene_name] / timing_data['samples_per_scene'][scene_name]
            }
            for scene_name in timing_data['scene_times']
        }
    }
    
    # Save timing log only if saving is enabled
    timing_file = None
    if save_results and version_dir is not None:
        timing_file = version_dir / "timing_log.json"
        with open(timing_file, 'w') as f:
            json.dump(timing_log, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE COMPLETE - PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Dataset version: {version}")
    print(f"Total samples processed: {timing_data['total_samples']}")
    print(f"Total time: {timing_data['total_time']:.2f} seconds")
    print(f"Average inference time per sample: {avg_inference_time:.4f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Min inference time: {np.min(timing_data['inference_times']):.4f}s")
    print(f"Max inference time: {np.max(timing_data['inference_times']):.4f}s")
    print(f"Std deviation: {np.std(timing_data['inference_times']):.4f}s")
    if save_results and base_dir is not None:
        print(f"Results saved to: {base_dir}")
        print(f"Timing log saved to: {timing_file}")
    else:
        print("No results saved to disk (--results flag not set)")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='FIERY Inference Testing')
    parser.add_argument('--checkpoint', required=True, type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', required=True, type=str, help='path to the dataset')
    parser.add_argument('--version', required=True, type=str, choices=['mini', 'trainval'],
                        help='dataset version')
    parser.add_argument('--results', action='store_true', help='save results to disk (default: False)')
    
    args = parser.parse_args()
    
    # Import cv2 here to avoid issues if not available
    global cv2
    import cv2
    
    run_inference_on_validation_set(args.checkpoint, args.dataroot, args.version, args.results)


if __name__ == '__main__':
    main()
