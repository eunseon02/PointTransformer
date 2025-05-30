import h5py

# HDF5 파일 열기
h5_filename = "lidar_data.h5"

with h5py.File(h5_filename, 'r') as h5_file:
    # 'valid/batch_0' 그룹 내의 데이터셋 나열
    if 'valid/batch_0' in h5_file:
        group = h5_file['valid/batch_0']
        print("Datasets in 'valid/batch_0':")
        for name in group.keys():
            print(name)
    else:
        print("'valid/batch_0' group not found in the HDF5 file.")
