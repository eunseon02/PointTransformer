import h5py
import re

# 수정 모드로 파일 열기
with h5py.File('final_dataset.h5', 'r+') as f:
    train_grp = f.get('train')
    if train_grp is None:
        raise KeyError("'train' 그룹이 파일에 없습니다!")

    # batch 이름 패턴 (batch_0000, batch_0001, ...)
    batch_pattern = re.compile(r'batch_\d+')

    # train 하위 그룹 순회
    for batch_name, batch_grp in train_grp.items():
        if not batch_pattern.fullmatch(batch_name):
            continue  # 혹시 다른 이름의 그룹이 있을 경우 건너뛰기

        # 삭제할 데이터셋 목록
        to_delete = []
        for ds_name in batch_grp:
            if ds_name in ('pts_0199', 'pts_0199_gt'):
                to_delete.append(ds_name)

        # 실제 삭제
        for ds_name in to_delete:
            print(f"Deleting: train/{batch_name}/{ds_name}")
            del batch_grp[ds_name]

    # 변경사항 저장
    f.flush()
