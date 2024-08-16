import time

# Example parameters
entry_bytes = 6547328  # 6.5 MB
batch_size = 32
num_gpu = 2

# Assume a hypothetical time per batch (you should measure this in your actual system)
time_per_batch = 0.1  # seconds

# Total data per batch
total_data_per_batch = entry_bytes * batch_size

# Throughput in bytes per second
throughput_bps = total_data_per_batch / time_per_batch

# Convert to GB/s
throughput_gbps = throughput_bps / (1024 ** 3)

print(f"Estimated GPU Throughput: {throughput_gbps:.3f} GB/s")
