import torch
from pathlib import Path
from diffusers import AutoencoderKL
import json
import random
import time
from PIL import Image
from PIL import ImageDraw, ImageFont

from SAFlow import SAFModel
from dataset import Stage1Dataset

def main() -> None:
    # 1) 加载配置
    with open("config.json", 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    vcfg = dict(cfg.get('verify', {}) or {})
    ckpt_path = Path(str(vcfg.get('checkpoint_path', ''))).expanduser()
    save_dir = Path(str(vcfg.get('save_dir', 'verify_outputs')))
    seed = int(vcfg.get('seed', 1234))
    num_steps = int(vcfg.get('num_steps', 20))
    cfg_scale = float(vcfg.get('cfg_scale', cfg.get('inference', {}).get('cfg_scale', 1.0)))
    sample_index = int(vcfg.get('sample_index', -1))

    if not ckpt_path.exists():
        raise FileNotFoundError(f"verify.checkpoint_path 不存在: {ckpt_path}")
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)

    # 2) 数据：各取一张 cls0/cls1 的 latent
    ds = Stage1Dataset(cfg['data']['data_root'], cfg['data'].get('num_classes'))
    if len(ds.classes) < 2:
        raise ValueError(f"需要至少 2 个类别，当前 len(classes)={len(ds.classes)}")

    def pick_latent_for_class(cls_id: int) -> torch.Tensor:
        files = ds.files_by_class[cls_id]
        if not files:
            raise RuntimeError(f"类别 {cls_id} 没有样本")
        if 0 <= sample_index < len(files):
            p = files[sample_index]
        else:
            p = rng.choice(files)
        x = ds.load_latent(p).unsqueeze(0)
        return x
    print(seed)
    x0 = pick_latent_for_class(0).to(device)
    x1 = pick_latent_for_class(1).to(device)

    # 3) 加载模型权重
    model = SAFModel(**cfg['model']).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 4) 加载 VAE（仅用于可视化）
    print("Loading VAE...")
    vae_dtype = torch.float16 if (device.type == 'cuda' and bool(vcfg.get('vae_fp16', True))) else torch.float32
    vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(device, dtype=vae_dtype)
    vae.eval()
    vae.requires_grad_(False)

    null_style_id = int(cfg.get('model', {}).get('num_styles', cfg['data'].get('num_classes', 0)))
    null_style = torch.tensor([null_style_id], device=device, dtype=torch.long)

    @torch.no_grad()
    def decode(latents: torch.Tensor) -> Image.Image:
        latents = latents.to(device)
        with torch.cuda.amp.autocast(enabled=(vae_dtype == torch.float16)):
            imgs = vae.decode(latents / 0.18215).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        arr = (imgs[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype('uint8')
        return Image.fromarray(arr)

    @torch.no_grad()
    def translate(x_c: torch.Tensor, target_id: int) -> torch.Tensor:
        x_t = x_c.clone()
        tid_tensor = torch.tensor([target_id], device=device, dtype=torch.long)
        # print(num_steps)
        for i in range(num_steps):
            t = torch.ones(1, device=device) * (i / float(num_steps))
            if cfg_scale > 1.0:
                x_in = torch.cat([x_t, x_t], dim=0)
                t_in = torch.cat([t, t], dim=0)
                c_in = torch.cat([x_c, x_c], dim=0)
                s_in = torch.cat([tid_tensor, null_style], dim=0)
                v_pred = model(x_in, c_in, t_in, s_in)
                v_cond, v_uncond = v_pred.chunk(2, dim=0)
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = model(x_t, x_c, t, tid_tensor)
            x_t = x_t + v * (1.0 / float(num_steps))
        return x_t

    # 5) 生成 6 张图（按你指定排版）
    # 左列：(cls0, 0to0, 1to0)
    # 右列：(cls1, 1to1, 0to1)
    img_cls0 = decode(x0)
    img_cls1 = decode(x1)
    img_0to0 = decode(translate(x0, 0))
    img_1to0 = decode(translate(x1, 0))
    img_1to1 = decode(translate(x1, 1))
    img_0to1 = decode(translate(x0, 1))

    # 6) 拼成一张大图
    w, h = img_cls0.size

    left_labels = ["cls0", "0to0", "1to0"]
    right_labels = ["cls1", "1to1", "0to1"]

    # 字体（优先 truetype，失败就用默认）
    font_size = int(vcfg.get('font_size', 20))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

    # 计算左右留白，保证文字不会被截断
    tmp = Image.new('RGB', (10, 10), color=(0, 0, 0))
    dtmp = ImageDraw.Draw(tmp)

    def text_width(s: str) -> int:
        try:
            box = dtmp.textbbox((0, 0), s, font=font)
            return int(box[2] - box[0])
        except Exception:
            return int(dtmp.textlength(s, font=font))

    pad = int(vcfg.get('label_pad', 12))
    left_margin = max([text_width(s) for s in left_labels] + [0]) + pad * 2
    right_margin = max([text_width(s) for s in right_labels] + [0]) + pad * 2

    canvas = Image.new('RGB', (left_margin + 2 * w + right_margin, 3 * h), color=(0, 0, 0))

    # 粘贴图片（整体右移 left_margin 给左侧标注腾位置）
    x0 = left_margin
    canvas.paste(img_cls0, (x0, 0))
    canvas.paste(img_0to0, (x0, h))
    canvas.paste(img_1to0, (x0, 2 * h))
    canvas.paste(img_cls1, (x0 + w, 0))
    canvas.paste(img_1to1, (x0 + w, h))
    canvas.paste(img_0to1, (x0 + w, 2 * h))

    # 画标注：左列在左侧留白区，右列在右侧留白区
    draw = ImageDraw.Draw(canvas)

    def draw_text_with_outline(x: int, y: int, s: str) -> None:
        # 黑色描边 + 白字，保证在各种图上都清晰
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((x + dx, y + dy), s, font=font, fill=(0, 0, 0))
        draw.text((x, y), s, font=font, fill=(255, 255, 255))

    for row in range(3):
        y_center = row * h + h // 2
        # 取近似文字高度
        try:
            box = draw.textbbox((0, 0), "Ag", font=font)
            th = int(box[3] - box[1])
        except Exception:
            th = int(font_size)

        y_text = int(y_center - th // 2)
        # 左列：靠近左图
        lx = int(pad)
        draw_text_with_outline(lx, y_text, left_labels[row])
        # 右列：靠近右图
        rx = int(left_margin + 2 * w + pad)
        draw_text_with_outline(rx, y_text, right_labels[row])

    ts = time.strftime('%Y%m%d-%H%M%S')
    out_name = f"verify_{ckpt_path.stem}_{seed}_{num_steps}_{ts}.jpg"
    out_path = save_dir / out_name
    canvas.save(out_path, quality=95)
    print(f"✅ 已保存: {out_path}")


if __name__ == '__main__':
    main()