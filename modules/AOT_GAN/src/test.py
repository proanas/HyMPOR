import os, glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from tqdm import tqdm
from utils.option import args
import importlib

def main_worker():
    net_module = importlib.import_module("model." + args.model)

    net = net_module.InpaintGenerator(args)
    net.load_state_dict(torch.load(args.pre_train, map_location="cpu"))
    net.eval().cuda()
    print(f"[✓] النموذج محمّل: {args.pre_train}")

    img_paths  = sorted(glob.glob(os.path.join(args.dir_image, "*.jpg"))
                      + glob.glob(os.path.join(args.dir_image, "*.png"))
                      + glob.glob(os.path.join(args.dir_image, "*.jpeg")))
    mask_paths = sorted(glob.glob(os.path.join(args.dir_mask,  "*.png"))
                      + glob.glob(os.path.join(args.dir_mask,  "*.jpg")))

    assert len(img_paths)  > 0, f"❌ لا توجد صور في: {args.dir_image}"
    assert len(mask_paths) > 0, f"❌ لا توجد أقنعة في: {args.dir_mask}"

    os.makedirs(args.outputs, exist_ok=True)
    print(f"[✓] {len(img_paths)} صورة | {len(mask_paths)} قناع")

    pairs = list(zip(img_paths, mask_paths * (len(img_paths) // len(mask_paths) + 1)))
    pairs = pairs[:len(img_paths)]

    for ipath, mpath in tqdm(pairs, desc="inpainting"):
        img  = Image.open(ipath).convert("RGB").resize((args.image_size, args.image_size))
        mask = Image.open(mpath).convert("L").resize((args.image_size,  args.image_size))

        img_t  = F.to_tensor(img).unsqueeze(0).cuda() * 2 - 1
        mask_t = (F.to_tensor(mask).unsqueeze(0).cuda() > 0.5).float()

        with torch.no_grad():
            pred = net(img_t * (1 - mask_t), mask_t)

        comp = (1 - mask_t) * img_t + mask_t * pred
        comp = ((comp + 1) / 2 * 255).clamp(0, 255)
        comp = comp.cpu().numpy()[0].transpose(1, 2, 0).astype(np.uint8)
        Image.fromarray(comp).save(os.path.join(args.outputs, os.path.basename(ipath)))

    print(f"[✓] تم الحفظ في: {args.outputs}")

if __name__ == "__main__":
    main_worker()
