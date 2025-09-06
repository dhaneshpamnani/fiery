#!/usr/bin/env python3

import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

def check_dataset_structure(dataroot, version):
    """Check the dataset structure and indexing logic"""
    
    print(f"=== Dataset Structure Analysis ===")
    print(f"Dataroot: {dataroot}")
    print(f"Version: {version}")
    print()
    
    # Load NuScenes dataset
    nusc = NuScenes(version=f'v1.0-{version}', dataroot=dataroot, verbose=False)
    
    # Get validation scenes
    splits = create_splits_scenes()
    val_scenes = splits[f'{version}_val']
    
    print(f"=== Validation Split Analysis ===")
    print(f"Total scenes in dataset: {len(nusc.scene)}")
    print(f"Total samples in dataset: {len(nusc.sample)}")
    print(f"Validation scenes: {len(val_scenes)}")
    print()
    
    print("Validation scene names:")
    for i, scene_name in enumerate(val_scenes):
        print(f"  {i}: {scene_name}")
    print()
    
    # Count samples per validation scene
    print("=== Samples per Validation Scene ===")
    val_scene_counts = {}
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        scene_name = nusc.get('scene', scene_token)['name']
        if scene_name in val_scenes:
            if scene_name not in val_scene_counts:
                val_scene_counts[scene_name] = 0
            val_scene_counts[scene_name] += 1
    
    total_val_samples = 0
    for scene_name in val_scenes:
        count = val_scene_counts.get(scene_name, 0)
        total_val_samples += count
        print(f"  {scene_name}: {count} samples")
    
    print(f"\nTotal validation samples: {total_val_samples}")
    print()
    
    # Analyze indexing logic
    print("=== Indexing Logic Analysis ===")
    print("Configuration:")
    print(f"  TIME_RECEPTIVE_FIELD (M): 3")
    print(f"  N_FUTURE_FRAMES (N): 4")
    print(f"  Total sequence length: 7")
    print()
    
    print("Expected sequences per scene:")
    for scene_name in val_scenes:
        sample_count = val_scene_counts.get(scene_name, 0)
        if sample_count > 0:
            # With M=3 past frames and N=4 future frames
            # First valid sequence starts at index M=3
            # Last valid sequence starts at index (total_samples - N - 1)
            # So we have: (total_samples - N) - M + 1 = total_samples - 6 sequences
            expected_sequences = max(0, sample_count - 6)  # 6 = M + N - 1
            print(f"  {scene_name}: {sample_count} samples -> {expected_sequences} sequences")
    
    print()
    print("=== Why So Many Scene Folders? ===")
    print("The issue is that each 'sample' in the dataset is actually a SEQUENCE of 7 frames.")
    print("The scene names you see are the scene tokens of the FIRST frame in each sequence.")
    print("Since sequences can start at different positions within a scene,")
    print("you get multiple 'scene' folders even though they're from the same actual scene.")
    print()
    
    # Show the actual scene mapping
    print("=== Actual Scene Mapping ===")
    actual_scenes = set()
    for sample in nusc.sample:
        scene_token = sample['scene_token']
        scene_name = nusc.get('scene', scene_token)['name']
        if scene_name in val_scenes:
            actual_scenes.add(scene_name)
    
    print(f"Actual validation scenes: {len(actual_scenes)}")
    for scene_name in sorted(actual_scenes):
        print(f"  {scene_name}")
    
    print()
    print("=== Summary ===")
    print(f"• There are {len(actual_scenes)} actual validation scenes")
    print(f"• Each scene has multiple samples (frames)")
    print(f"• The dataset creates sequences of 7 consecutive frames")
    print(f"• Each sequence gets labeled with its starting scene")
    print(f"• This creates many 'scene' folders, but they're actually sequences from the same scenes")

def main():
    parser = argparse.ArgumentParser(description='Check dataset structure and indexing')
    parser.add_argument('--dataroot', required=True, help='Path to dataset')
    parser.add_argument('--version', required=True, choices=['mini', 'trainval'], help='Dataset version')
    
    args = parser.parse_args()
    check_dataset_structure(args.dataroot, args.version)

if __name__ == '__main__':
    main()
