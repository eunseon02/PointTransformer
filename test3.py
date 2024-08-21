import torch
import spconv.pytorch as spconv
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


batch_size = 2
point_cloud_shape = (batch_size, 1000, 3)
random_pc = torch.rand(point_cloud_shape, dtype=torch.float32)
all_voxels, all_indices = [], []
input_shape = (50, 120, 120)
for batch_idx in range(batch_size):
    # 원래 코드는 여기서 pc_single을 기반으로 point_to_voxel_hash를 사용했지만
    # 무작위 텐서를 사용하여 그 부분을 대체합니다.

    # voxels_tv와 indices_tv를 랜덤하게 생성
    num_voxels = 2000  # 테스트용 임의의 값, 원래 데이터셋 크기에 따라 조정 가능
    num_point_features = 3
    voxels_torch = torch.rand((num_voxels, num_point_features), dtype=torch.float32).to(device)
    indices_torch = torch.randint(0, input_shape[0], (num_voxels, 3), dtype=torch.int32).to(device)

    print("NaN values in voxels_torch:", torch.isnan(voxels_torch).any())
    print("Inf values in voxels_torch:", torch.isinf(voxels_torch).any())
    print("NaN values in indices_torch:", torch.isnan(indices_torch).any())
    print("Inf values in indices_torch:", torch.isinf(indices_torch).any())

    # 무작위 인덱스와 배치를 결합
    batch_indices = torch.full((indices_torch.shape[0], 1), batch_idx, dtype=torch.int32).to(device)
    indices_combined = torch.cat([batch_indices, indices_torch], dim=1)

    print(indices_combined)
    print(voxels_torch)

    all_voxels.append(voxels_torch)
    all_indices.append(indices_combined.int())

all_voxels = torch.cat(all_voxels, dim=0)
all_indices = torch.cat(all_indices, dim=0)
sparse_tensor = spconv.SparseConvTensor(all_voxels, all_indices, input_shape, batch_size)

print("input", input_shape)
print("SparseConvTensor features shape:", sparse_tensor.features.shape)
print("SparseConvTensor indices shape:", sparse_tensor.indices.shape)
print("SparseConvTensor spatial shape:", sparse_tensor.spatial_shape)

dense_tensor = sparse_tensor.dense()
print("Dense Tensor shape:", dense_tensor.shape)
print("NaN values in tensor:", torch.isnan(dense_tensor).any())
print("Inf values in tensor:", torch.isinf(dense_tensor).any())
print("Dense Tensor shape:", dense_tensor[:10])
