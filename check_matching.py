import os

def get_non_matching_files(directory):
    # 디렉토리 내의 모든 파일을 가져옵니다.
    all_files = os.listdir(directory)
    
    # '_gt.ply'가 포함된 파일과 포함되지 않은 파일을 구분합니다.
    data_files = [f for f in all_files if not f.endswith('_gt.ply') and f.endswith('.ply')]
    gt_files = [f for f in all_files if f.endswith('_gt.ply')]
    csv_file = [f for f in all_files if f.endswith('.csv')]

    # 매칭되지 않는 파일 쌍을 찾습니다.
    non_matching_pairs = []
    for data_file in data_files:
        gt_file = data_file.replace('.ply', '_gt.ply')
        if gt_file not in gt_files:
            non_matching_pairs.append((os.path.join(directory, data_file), None))

    for gt_file in gt_files:
        data_file = gt_file.replace('_gt.ply', '.ply')
        if data_file not in data_files:
            non_matching_pairs.append((None, os.path.join(directory, gt_file)))

    return non_matching_pairs, len(data_files), len(gt_files),csv_file

def check_all_batches(root_dir):
    total_data_files = 0
    total_gt_files = 0
    for batch_folder in sorted(os.listdir(root_dir)):
        batch_dir = os.path.join(root_dir, batch_folder)
        if os.path.isdir(batch_dir):
            print(f"Checking batch directory: {batch_dir}")
            non_matching_files, data_file_count, gt_file_count, csv_file = get_non_matching_files(batch_dir)
            total_data_files += data_file_count
            total_gt_files += gt_file_count
            for data_file, gt_file in non_matching_files:
                if data_file:
                    print(f"Data file without matching GT file: {data_file}")
                if gt_file:
                    print(f"GT file without matching data file: {gt_file}")
            print(f"csv:{csv_file}")

    print(f"Total data files: {total_data_files}")
    print(f"Total GT files: {total_gt_files}")

# 예제 사용
root_dir = 'dataset2/valid'
check_all_batches(root_dir)
