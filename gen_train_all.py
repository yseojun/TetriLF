import numpy as np
import os
import json

# =============================================================================
# 데이터셋별 프레임 수 (여기서 수정)
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

# =============================================================================
# 학습할 데이터셋 이름들 (여기서 수정)
# =============================================================================
DATASETS = list(DATASET_FRAMES.keys())  # 전체 학습시 사용
# 또는 특정 데이터셋만 학습하려면 아래처럼 직접 지정:
# DATASETS = ['ambushfight_1', 'bamboo_1']

# =============================================================================
# 공통 설정
# =============================================================================
DATA_ROOT = '/data/ysj/dataset/LF_video_half'
LOG_ROOT = '/data/ysj/result/tetrirf/logs/0108_4_64_half'
TEMPLATE_CONFIG = 'configs/LF/template_lf.py'
EXP_SUFFIX = '0108_4_64_half'

# 학습 설정
STEP = 10
OVERLAP = True


def generate_config(dataset_name):
    """데이터셋별 config 파일 생성"""
    config_path = f'configs/LF/{dataset_name}_lf.py'
    
    # 템플릿 읽기
    with open(TEMPLATE_CONFIG, 'r') as f:
        template = f.read()
    
    # 값 치환
    expname = f'lf_{dataset_name}_{EXP_SUFFIX}'
    datadir = f'{DATA_ROOT}/{dataset_name}'
    
    config_content = template.replace(
        "expname = 'TEMPLATE'",
        f"expname = '{expname}'"
    ).replace(
        "basedir = 'TEMPLATE'",
        f"basedir = '{LOG_ROOT}'"
    ).replace(
        "datadir='TEMPLATE'",
        f"datadir='{datadir}'"
    )
    
    # config 파일 저장
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"Generated config: {config_path}")
    return config_path


def generate_train_commands(config_file, dataset_name):
    """학습 명령어들 생성"""
    total_frames = DATASET_FRAMES[dataset_name]
    commands = []
    
    for i in range(0, total_frames, STEP):
        if i != 0 and OVERLAP:
            start = i - 1
        else:
            start = i
        end = min(i + STEP, total_frames)  # 프레임 수 초과 방지
        
        tmp = " ".join([str(j) for j in range(start, end)])
        
        mode = 0 if i == 0 else 1
        cmd = f"python run_multiframe.py --config {config_file} --frame_ids {tmp} --training_mode {mode}"
        commands.append(cmd)
    
    return commands


def main():
    # 전체 실행 스크립트 생성
    all_commands = []
    
    for dataset_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print('='*60)
        
        # 1. config 파일 생성
        config_file = generate_config(dataset_name)
        
        # 2. 학습 명령어 생성 (데이터셋별 프레임 수 적용)
        commands = generate_train_commands(config_file, dataset_name)
        all_commands.extend(commands)
        
        print(f"  Total frames: {DATASET_FRAMES[dataset_name]}")
        
        # 구분을 위한 빈 줄
        all_commands.append(f"echo '=== Finished {dataset_name} ==='")
        all_commands.append("")
    
    # run_all.sh 파일 생성
    filename = './run_all.sh'
    with open(filename, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script to train all LF datasets\n\n")
        for cmd in all_commands:
            f.write(cmd + "\n")
    
    print(f"\n{'='*60}")
    print(f"Generated: {filename}")
    print(f"Total datasets: {len(DATASETS)}")
    print(f"Run with: sh {filename}")
    print('='*60)
    
    # 실행 여부 확인
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--run':
        print("\nStarting training...")
        os.system(f"sh {filename}")
    else:
        print("\nTo run: python gen_train_all.py --run")
        print("Or manually: sh run_all.sh")


if __name__ == '__main__':
    main()

