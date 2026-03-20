import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

from config import system_configs
from db.datasets import datasets
from nnet.py_factory import NetworkFactory
from utils import normalize_


def main():
    cfg_name = 'LSTR_CULANE_2k_mamba_dense'
    cfg_path = Path('config') / f'{cfg_name}.json'
    with cfg_path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)

    cfg['system']['snapshot_name'] = cfg_name
    system_configs.update_config(cfg['system'])

    nnet = NetworkFactory(flag=False)
    nnet.load_params(12500)
    nnet.cuda()
    nnet.eval_mode()

    db = datasets['CULANE'](cfg['db'], 'test')
    scores_all = []

    num_images = min(500, len(db.db_inds))
    for ind in range(num_images):
        db_ind = db.db_inds[ind]
        image = cv2.imread(db.image_file(db_ind))
        if image is None:
            continue

        images_np = np.zeros((1, 3, 295, 820), dtype=np.float32)
        # Correct repo semantics: 0=valid, 1=invalid/padded.
        masks_np = np.zeros((1, 1, 295, 820), dtype=np.float32)

        resized = cv2.resize(image, (820, 295)) / 255.0
        normalize_(resized, db.mean, db.std)
        images_np[0] = resized.transpose(2, 0, 1)

        images_t = torch.from_numpy(images_np).cuda()
        masks_t = torch.from_numpy(masks_np).cuda()

        with torch.no_grad():
            outputs, _ = nnet.test([images_t, masks_t])

        fg_probs = F.softmax(outputs['pred_object_logits'], dim=-1)[..., 0]
        scores_all.extend(fg_probs[0].detach().cpu().tolist())

    if not scores_all:
        raise RuntimeError('No scores collected; check dataset paths and checkpoint loading.')

    scores_all = sorted(scores_all)
    total = len(scores_all)
    denom_images = max(num_images, 1)

    print('=== SCORE DAGILIMI (500 goruntuye kadar, 7 query/goruntu) ===')
    print(f'Toplam query: {total}')
    print(f'Medyan: {np.median(scores_all):.3f}')
    print(f'Mean:   {np.mean(scores_all):.3f}')
    print()

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n = sum(1 for s in scores_all if s >= thresh)
        pct = 100.0 * n / total
        avg_per_img = n / denom_images
        print(f'thresh={thresh:.1f}: {n:4d}/{total} ({pct:5.1f}%) -> ortalama {avg_per_img:.2f} lane/goruntu')

    print()
    print('CULane gercegi: ortalama ~2-3 lane/goruntu')
    print('Hangisi buna yakin? O threshold kullan.')


if __name__ == '__main__':
    main()
