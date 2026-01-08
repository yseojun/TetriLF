"""
Light Field Compression/Decompression Automation Script

This script runs compression_lf.py and videos_to_planes_lf.py for all trained datasets.
Usage: python gen_compress_all.py --dir /path/to/logs --qp 20 [--run]
"""

import os
import sys
import argparse
import re

# =============================================================================
# 데이터셋별 프레임 수 (gen_train_all.py에서 복사)
# =============================================================================
DATASET_FRAMES = {
    # ambushfight: 20 frames
    'ambushfight_1': 20,
    'ambushfight_2': 20,
    'ambushfight_3': 20,
    'ambushfight_4': 20,
    'ambushfight_5': 20,
    'ambushfight_6': 20,
    # bamboo: 50 frames
    'bamboo_1': 50,
    'bamboo_2': 50,
    'bamboo_3': 50,
    # chickenrun: 50 frames
    'chickenrun_1': 50,
    'chickenrun_2': 50,
    'chickenrun_3': 50,
    # foggyrocks: 50 frames
    'foggyrocks_1': 50,
    'foggyrocks_2': 50,
    # questbegins: 40 frames
    'questbegins_1': 40,
    # shaman: 50 frames
    'shaman_1': 50,
    'shaman_2': 50,
    'shaman_3': 50,
    # shaman_b: 48 frames
    'shaman_b_1': 48,
    'shaman_b_2': 48,
    # thebigfight: 50 frames
    'thebigfight_1': 50,
    'thebigfight_2': 50,
    'thebigfight_3': 50,
    'thebigfight_4': 50,
}


def extract_dataset_name(folder_name):
    """
    폴더 이름에서 데이터셋 이름 추출
    예: 'lf_ambushfight_1_0107_4_64_half' -> 'ambushfight_1'
    """
    # lf_ 접두사 제거
    if folder_name.startswith('lf_'):
        folder_name = folder_name[3:]
    
    # DATASET_FRAMES에 있는 키와 매칭
    for dataset_name in DATASET_FRAMES.keys():
        if folder_name.startswith(dataset_name + '_') or folder_name == dataset_name:
            return dataset_name
    
    return None


def find_trained_folders(base_dir):
    """
    base_dir 하위에서 학습 완료된 폴더들을 찾음
    lf_{dataset_name}_{suffix} 패턴의 폴더를 찾음
    """
    trained_folders = []
    
    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        return trained_folders
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
        
        # lf_ 접두사로 시작하는 폴더만 처리
        if not folder_name.startswith('lf_'):
            continue
        
        # 데이터셋 이름 추출
        dataset_name = extract_dataset_name(folder_name)
        if dataset_name is None:
            print(f"Warning: Unknown dataset pattern: {folder_name}")
            continue
        
        # fine_last_0.tar 파일이 있는지 확인 (학습 완료 여부)
        if not os.path.isfile(os.path.join(folder_path, 'fine_last_0.tar')):
            print(f"Warning: No checkpoint found in {folder_name}, skipping...")
            continue
        
        num_frames = DATASET_FRAMES[dataset_name]
        trained_folders.append({
            'path': folder_path,
            'name': folder_name,
            'dataset': dataset_name,
            'num_frames': num_frames
        })
    
    return trained_folders


def generate_compress_commands(trained_folders, qp, codec):
    """압축 명령어 생성"""
    commands = []
    
    for folder_info in trained_folders:
        folder_path = folder_info['path']
        num_frames = folder_info['num_frames']
        dataset_name = folder_info['dataset']
        
        # 압축 명령어
        compress_cmd = (
            f"python tools/compression_lf.py "
            f"--logdir {folder_path} "
            f"--numframe {num_frames} "
            f"--qp {qp} "
            f"--codec {codec}"
        )
        commands.append(compress_cmd)
        
        # 복원 명령어
        compressed_dir = os.path.join(folder_path, f'compressed_{qp}')
        decompress_cmd = (
            f"python tools/videos_to_planes_lf.py "
            f"--dir {compressed_dir} "
            f"--numframe {num_frames} "
            f"--codec {codec} "
            f"--no_wandb"
        )
        commands.append(decompress_cmd)
        
        # 구분을 위한 로그
        commands.append(f"echo '=== Finished {dataset_name} compression/decompression ==='")
        commands.append("")
    
    return commands


def main():
    parser = argparse.ArgumentParser(
        description='Compress and decompress all trained LF datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dir', required=True,
                        help='Base directory containing trained model folders')
    parser.add_argument('--qp', type=int, default=20,
                        help='QP value for video codec compression')
    parser.add_argument('--codec', type=str, default='h265',
                        choices=['h265', 'mpg2'],
                        help='Video codec to use')
    parser.add_argument('--run', action='store_true',
                        help='Run the generated script immediately')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Light Field Compression/Decompression Tool")
    print(f"{'='*60}")
    print(f"  Base Directory: {args.dir}")
    print(f"  QP: {args.qp}")
    print(f"  Codec: {args.codec}")
    print(f"{'='*60}\n")
    
    # 학습 완료된 폴더 찾기
    trained_folders = find_trained_folders(args.dir)
    
    if not trained_folders:
        print("No trained folders found!")
        return
    
    print(f"Found {len(trained_folders)} trained folders:")
    for folder_info in trained_folders:
        print(f"  - {folder_info['name']} ({folder_info['num_frames']} frames)")
    print()
    
    # 명령어 생성
    commands = generate_compress_commands(trained_folders, args.qp, args.codec)
    
    # 스크립트 파일 생성
    script_filename = f'./run_compress_all_qp{args.qp}.sh'
    with open(script_filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated script to compress/decompress all LF datasets\n")
        f.write(f"# QP: {args.qp}, Codec: {args.codec}\n\n")
        for cmd in commands:
            f.write(cmd + "\n")
    
    print(f"Generated: {script_filename}")
    print(f"Total datasets: {len(trained_folders)}")
    
    if args.run:
        print("\nStarting compression/decompression...")
        os.system(f"bash {script_filename}")
    else:
        print(f"\nTo run: python gen_compress_all.py --dir {args.dir} --qp {args.qp} --run")
        print(f"Or manually: bash {script_filename}")


if __name__ == '__main__':
    main()

