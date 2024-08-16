# Given parameters
entry_bytes = 6547328  # bytes per entry
batch_size = 32
num_gpu = 2
gpu_throughput_gbps = 1.951  # GB/s

# Calculate batch data size in bytes and GB
batch_data_bytes = entry_bytes * batch_size
batch_data_gb = batch_data_bytes / (1024 ** 3)

# Calculate time per batch
time_per_batch = batch_data_gb / gpu_throughput_gbps

# Calculate the number of workers
num_workers = (gpu_throughput_gbps * num_gpu) / (entry_bytes * batch_size / (1024 ** 3))

print(f"Estimated Number of Workers: {int(num_workers)}")
