"""
Light Field Rendering Test Automation Script

This script generates render_lf.py commands for all trained datasets.
Usage: python gen_test_all.py [--reald] [--qp 20] [--run]
"""

import os
import sys
import argparse

# =============================================================================
# 데이터셋별 프레임 수 (gen_train_all.py에서 복사)
# =============================================================================
DATASET_FRAMES = {
    'ambushfight_1': 20,
    'ambushfight_2': 21,
    'ambushfight_3': 41,
    'ambushfight_4': 30,
    'ambushfight_5': 50,
    'ambushfight_6': 20,
    
    'bamboo_1': 50,
    'bamboo_2': 50,
    'bamboo_3': 50,
    
    'chickenrun_1': 50,
    'chickenrun_2': 21,
    'chickenrun_3': 50,
    
    'foggyrocks_1': 50,
    'foggyrocks_2': 50,
    
    'questbegins_1': 40,
    
    'shaman_1': 50,
    'shaman_2': 50,
    'shaman_3': 50,
    
    'shaman_b_1': 48,
    'shaman_b_2': 50,
    
    'thebigfight_1': 50,
    'thebigfight_2': 50,
    'thebigfight_3': 50,
    'thebigfight_4': 50,
}

# =============================================================================
# 테스트할 데이터셋 이름들 (여기서 수정)
# =============================================================================
DATASETS = list(DATASET_FRAMES.keys())  # 전체 테스트시 사용
# 또는 특정 데이터셋만 테스트하려면 아래처럼 직접 지정:
# DATASETS = ['ambushfight_1', 'bamboo_1']

# =============================================================================
# 공통 설정 (gen_train_all.py와 동일하게 설정)
# =============================================================================

EXP_SUFFIX = '0103_4_128'

LOG_ROOT = '/data/ysj/result/tetrirf/logs/' + EXP_SUFFIX


def generate_test_commands(dataset_name, step, reald=False, qp=None, codec='h265'):
    """렌더링 테스트 명령어 생성"""
    total_frames = DATASET_FRAMES[dataset_name]
    # 해당 데이터셋의 학습 결과 폴더에 있는 config.py 사용
    expname = f'lf_{dataset_name}_{EXP_SUFFIX}'
    config_file = f'{LOG_ROOT}/{expname}/config.py'
    commands = []
    
    for i in range(0, total_frames, step):
        end = min(i + step, total_frames)
        frame_ids = " ".join([str(j) for j in range(i, end)])
        
        if reald:
            # 원본 데이터 렌더링
            cmd = (
                f"python render_lf.py --config {config_file} "
                f"--frame_ids {frame_ids} --render_only --render_test --reald"
            )
        else:
            # 압축 데이터 렌더링
            cmd = (
                f"python render_lf.py --config {config_file} "
                f"--frame_ids {frame_ids} --render_only --render_test "
                f"--qp {qp} --codec {codec}"
            )
        commands.append(cmd)
    
    return commands


def main():
    parser = argparse.ArgumentParser(
        description='Generate rendering test commands for all LF datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--reald', action='store_true',
                        help='Use original (uncompressed) data for rendering')
    parser.add_argument('--qp', type=int, default=20,
                        help='QP value for compressed data rendering (ignored if --reald)')
    parser.add_argument('--codec', type=str, default='h265',
                        choices=['h265', 'mpg2'],
                        help='Codec used for compression (ignored if --reald)')
    parser.add_argument('--step', type=int, default=10,
                        help='Number of frames to render per batch')
    parser.add_argument('--datasets', nargs='+', type=str, default=None,
                        help='Specific datasets to test (default: all)')
    parser.add_argument('--run', action='store_true',
                        help='Run the generated script immediately')
    
    args = parser.parse_args()
    
    # 데이터셋 선택
    datasets_to_test = args.datasets if args.datasets else DATASETS
    
    # 유효한 데이터셋만 필터링
    valid_datasets = []
    for ds in datasets_to_test:
        if ds in DATASET_FRAMES:
            valid_datasets.append(ds)
        else:
            print(f"Warning: Unknown dataset '{ds}', skipping...")
    
    if not valid_datasets:
        print("No valid datasets to test!")
        return
    
    render_type = "원본 데이터 (--reald)" if args.reald else f"압축 데이터 (QP={args.qp}, codec={args.codec})"
    
    print(f"\n{'='*60}")
    print(f"  Light Field Rendering Test Generator")
    print(f"{'='*60}")
    print(f"  Render Type: {render_type}")
    print(f"  Step Size: {args.step}")
    print(f"  Datasets: {len(valid_datasets)}")
    print(f"{'='*60}\n")
    
    # 전체 명령어 생성
    all_commands = []
    
    for dataset_name in valid_datasets:
        print(f"Processing: {dataset_name} ({DATASET_FRAMES[dataset_name]} frames)")
        
        commands = generate_test_commands(
            dataset_name, 
            args.step, 
            reald=args.reald, 
            qp=args.qp, 
            codec=args.codec
        )
        all_commands.extend(commands)
        
        # 구분을 위한 로그
        all_commands.append(f"echo '=== Finished {dataset_name} rendering ==='")
        all_commands.append("")
    
    # 스크립트 파일 생성
    if args.reald:
        script_filename = './run_test_all_reald.sh'
    else:
        script_filename = f'./run_test_all_qp{args.qp}.sh'
    
    with open(script_filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Auto-generated script to render all LF datasets\n")
        if args.reald:
            f.write(f"# Render Type: Original data (--reald)\n\n")
        else:
            f.write(f"# Render Type: Compressed data (QP={args.qp}, codec={args.codec})\n\n")
        
        for cmd in all_commands:
            f.write(cmd + "\n")
    
    print(f"\n{'='*60}")
    print(f"Generated: {script_filename}")
    print(f"Total datasets: {len(valid_datasets)}")
    print(f"Total commands: {len([c for c in all_commands if c.startswith('python')])}")
    print(f"{'='*60}")
    
    if args.run:
        print("\nStarting rendering...")
        os.system(f"bash {script_filename}")
    else:
        print(f"\nTo run: python gen_test_all.py {'--reald' if args.reald else f'--qp {args.qp}'} --run")
        print(f"Or manually: bash {script_filename}")


if __name__ == '__main__':
    main()

