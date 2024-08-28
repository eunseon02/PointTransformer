




### **Requirements**

1. install spconv

```bash
pip install spconv-cu118	
```

1. install chamfer loss 

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install git+'https://github.com/otaheri/chamfer_distance'
```


### **Train**


```bash
python train_spconv.py
```


memo
```python
output_directory = "train_"
file_path = os.path.join(output_directory, f'{iter}.joblib')
print(f"iter : {iter}, len_occu : {len(self.train_occu)}")
if iter >= len(self.train_occu): 
    if os.path.exists(file_path):
        self.train_occu.append(joblib.load(file_path, mmap_mode='r'))
        print("File loaded successfully.")
        # if occupancy_grids[0].size(0) != self.batch_size:
        #     print("error")
    else:
        print(f"File '{file_path}' does not exist.")
        occupancy_grids = []
        # torch.tensor([5, 14, 14], dtype=torch.float32)
        occupancy_grids.append(self.occupancy_grid(gt_pts, (5, 14, 14), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([5, 14, 14], dtype=torch.float32)))
        occupancy_grids.append(self.occupancy_grid(gt_pts, (11, 29, 29), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([11, 29, 29], dtype=torch.float32)))
        occupancy_grids.append(self.occupancy_grid(gt_pts, (24, 59, 59), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([24, 59, 59], dtype=torch.float32)))
        occupancy_grids.append(self.occupancy_grid(gt_pts, (50, 120, 120), (self.max_coord_range_xyz - self.min_coord_range_xyz) / torch.tensor([50, 120, 120], dtype=torch.float32)))
        os.makedirs(output_directory, exist_ok=True)
        joblib.dump(occupancy_grids, file_path)
        self.train_occu.append(joblib.load(file_path, mmap_mode='r'))
occupancy_grids = self.train_occu[iter] 
```
RAM사용 너무 큼
