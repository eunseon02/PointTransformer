import torch
import spconv.pytorch as spconv

# 랜덤 텐서 생성
num_points = 2048
num_channels = 3
spatial_shape = [50, 120, 120]
batch_size = 1

# features: [N, num_channels]
features = torch.rand(num_points, num_channels).to(torch.float32)

# indices: [N, ndim + 1] -> (batch_idx, z, y, x)
indices = torch.cat([
    torch.zeros((num_points, 1), dtype=torch.int32),  # batch_idx
    torch.randint(0, spatial_shape[0], (num_points, 1), dtype=torch.int32),  # z
    torch.randint(0, spatial_shape[1], (num_points, 1), dtype=torch.int32),  # y
    torch.randint(0, spatial_shape[2], (num_points, 1), dtype=torch.int32)   # x
], dim=1)

# SparseConvTensor 생성
sparse_tensor = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)

# Dense tensor로 변환
dense_tensor = sparse_tensor.dense()

print(dense_tensor)
# 결과 출력
print(f"SparseConvTensor features shape: {features.shape}")
print(f"SparseConvTensor indices shape: {indices.shape}")
print(f"SparseConvTensor spatial shape: {spatial_shape}")
print(f"Dense Tensor shape: {dense_tensor.shape}")
