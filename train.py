import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from pathlib import Path
import json, random, re, shutil, os, time
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL
try:
    from transformers import get_cosine_schedule_with_warmup  # type: ignore
except Exception:
    get_cosine_schedule_with_warmup = None  # type: ignore
from typing import Dict, List, Optional, Tuple

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore

try:
    from torch_ema import ExponentialMovingAverage  # type: ignore
except ImportError:
    ExponentialMovingAverage = None  # type: ignore

from SAFlow import SAFModel
from dataset import Stage1Dataset, Stage2Dataset

# ================= 核心参数 =================
NUM_SAMPLES_PER_CLASS = 1  # 推理时每个类别选几张图
IDENTITY_RATE = 0.2        # 20% 概率学 Identity
EVAL_STEP = 1              # 每个 Epoch 结束后保存并推理
# ===========================================

class ReflowTrainer:
    def __init__(self):
        print("\n🔧 [Init] 初始化训练器...")
        with open("config.json", 'r', encoding='utf-8') as f:
            self.cfg = json.load(f)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ckpt_dir = Path(self.cfg['checkpoint']['save_dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.reflow_dir = Path(self.cfg['training']['reflow_data_dir'])
        # 修改保存路径为当前目录下的 visualizations 文件夹
        self.vis_root = Path("visualizations")

        # TensorBoard
        tb_cfg = dict(self.cfg.get('training', {}).get('tensorboard', {}) or {})
        self.tb_enabled = bool(tb_cfg.get('enabled', True))
        self.tb_log_every = int(tb_cfg.get('log_every', 10))
        self.tb_flush_secs = int(tb_cfg.get('flush_secs', 5))
        self.tb_log_dir: Optional[Path] = None
        self.tb_writer = None
        if self.tb_enabled:
            if SummaryWriter is None:
                raise ImportError("需要安装 tensorboard 才能使用 TensorBoard 可视化。请先执行: pip install tensorboard")
            log_root = Path(tb_cfg.get('log_dir', str(self.ckpt_dir / 'tb_runs')))
            run_name = str(tb_cfg.get('run_name', time.strftime('%Y%m%d-%H%M%S')))
            self.tb_log_dir = log_root / run_name
            self.tb_log_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(self.tb_log_dir), flush_secs=self.tb_flush_secs)
            # 记录配置，方便复现实验
            try:
                (self.tb_log_dir / 'config.json').write_text(json.dumps(self.cfg, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception:
                pass

        # tqdm 显示控制：固定宽度避免终端缩放导致频繁重绘
        tqdm_cfg = dict(self.cfg.get('training', {}).get('tqdm', {}) or {})
        self.tqdm_ncols = int(tqdm_cfg.get('ncols', 120))
        self.tqdm_dynamic_ncols = bool(tqdm_cfg.get('dynamic_ncols', False))

        # 监控/可视化频率：避免每步 .item() 触发 GPU 同步导致“卡住”观感
        self.monitor_every = int(self.cfg.get('training', {}).get('monitor_every', 10))
        self.eval_step = int(self.cfg.get('training', {}).get('eval_step', EVAL_STEP))
        self.do_infer = bool(self.cfg.get('training', {}).get('do_inference', True))
        self.keep_vae_on_gpu = bool(self.cfg.get('training', {}).get('keep_vae_on_gpu', False))

        # AMP + GradScaler（开启 use_amp 时启用）
        self.use_amp = bool(self.cfg.get('training', {}).get('use_amp', False)) and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # 梯度防爆配置
        grad_cfg = dict(self.cfg.get('training', {}).get('grad', {}) or {})
        self.grad_clip_max_norm = float(grad_cfg.get('clip_max_norm', 1.0))
        self.grad_skip_norm_threshold = float(grad_cfg.get('skip_norm_threshold', 0.0))
        self.grad_skip_on_nonfinite_loss = bool(grad_cfg.get('skip_on_nonfinite_loss', True))

        # 数据尺度自检（防止把 0-255 像素当 latent 喂进去）
        dsc = dict(self.cfg.get('training', {}).get('data_sanity_check', {}) or {})
        self.data_check_enabled = bool(dsc.get('enabled', True))
        self.data_check_absmean_threshold = float(dsc.get('absmean_threshold', 10.0))
        self.data_check_max_threshold = float(dsc.get('max_threshold', 10.0))
        self.data_check_print_once = bool(dsc.get('print_once', True))
        self.data_check_auto_fix = bool(dsc.get('auto_fix', False))
        self.data_check_fix_mode = str(dsc.get('fix_mode', '0_255_to_minus1_1')).lower().strip()
        self._data_alert_printed = False
        
        # 加载 VAE (冻结)
        print("⏳ [Init] 加载 VAE...")
        # 训练全程只用 latent；VAE 仅用于可视化，因此默认放在 CPU 防止占用显存
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.vae.eval()
        self.vae.requires_grad_(False)

        # 准备类别权重张量
        self.id_loss_weights = torch.tensor(
            self.cfg['training'].get('identity_weights', [1.0] * self.cfg['data']['num_classes']),
            device=self.device
        )

        # Style Dropout（Classifier-Free Style Guidance 思路）：以一定概率将 style_id 置为 null_style_id
        sd_cfg = dict(self.cfg.get('training', {}).get('style_dropout', {}) or {})
        self.style_dropout_enabled = bool(sd_cfg.get('enabled', True))
        self.style_dropout_p = float(sd_cfg.get('p', 0.1))
        self.style_dropout_per_sample = bool(sd_cfg.get('per_sample', True))
        self.null_style_id = int(self.cfg.get('model', {}).get('num_styles', self.cfg['data'].get('num_classes', 0)))

        # Gram Matrix 风格损失（默认在 latent 空间做，避免与内容强绑定）
        gram_cfg = dict(self.cfg.get('training', {}).get('gram_loss', {}) or {})
        self.gram_enabled = bool(gram_cfg.get('enabled', False))
        self.gram_lambda = float(gram_cfg.get('lambda', 0.0))
        self.gram_apply_on = str(gram_cfg.get('apply_on', 'style_only'))  # 'style_only' | 'all'

        # EMA
        ema_cfg = dict(self.cfg.get('training', {}).get('ema', {}) or {})
        self.ema_enabled = bool(ema_cfg.get('enabled', True))
        self.ema_decay = float(ema_cfg.get('decay', 0.999))

        if self.ema_enabled and ExponentialMovingAverage is None:
            raise ImportError("需要安装 torch-ema 才能使用 EMA。请先执行: pip install torch-ema")

    def close(self) -> None:
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
            except Exception:
                pass
            try:
                self.tb_writer.close()
            except Exception:
                pass

    def _apply_style_dropout(self, style_id: torch.Tensor) -> torch.Tensor:
        if (not self.style_dropout_enabled) or self.style_dropout_p <= 0:
            return style_id
        if self.style_dropout_p >= 1:
            return torch.full_like(style_id, self.null_style_id)

        if self.style_dropout_per_sample:
            # per-sample Bernoulli mask
            mask = torch.rand(style_id.shape, device=style_id.device) < self.style_dropout_p
            if not mask.any():
                return style_id
            out = style_id.clone()
            out[mask] = self.null_style_id
            return out

        # per-batch dropout
        if random.random() < self.style_dropout_p:
            return torch.full_like(style_id, self.null_style_id)
        return style_id

    @staticmethod
    def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
        """Gram matrix for style loss.

        feat: [B, C, H, W] -> gram: [B, C, C]
        """
        if feat.ndim != 4:
            raise ValueError(f"gram_matrix expects 4D tensor, got shape={tuple(feat.shape)}")
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        gram = torch.bmm(f, f.transpose(1, 2))
        gram = gram / float(c * h * w)
        return gram

    def _gram_style_loss(self, pred_feat: torch.Tensor, target_feat: torch.Tensor) -> torch.Tensor:
        """Per-sample Gram style loss in latent space. Returns shape [B]."""
        # 用 float32 计算更稳（AMP 下避免数值问题）
        pred = pred_feat.float()
        target = target_feat.float()
        g1 = self._gram_matrix(pred)
        g2 = self._gram_matrix(target)
        return torch.mean((g1 - g2) ** 2, dim=[1, 2])

    def _monitor_cfg(self, section_cfg: Dict, *, default_method: str) -> Dict:
        """读取“监工过滤”配置。

        放在 training.filter_wo_decode.monitor / training.filter_w_decode.monitor 下。
        """
        monitor = dict(section_cfg.get('monitor', {}) or {})
        enabled = bool(monitor.get('enabled', True))
        ratio = float(monitor.get('ratio', 0.1))
        num_pairs = int(monitor.get('num_pairs', 128))
        style_a = int(monitor.get('style_a', 0))
        style_b = int(monitor.get('style_b', 1))
        method = str(monitor.get('method', default_method))
        allow_empty = bool(monitor.get('allow_empty', False))
        min_threshold = float(monitor.get('min_threshold', 0.0))
        batch_size = int(monitor.get('batch_size', 8))
        seed = int(monitor.get('seed', 1234))
        return {
            'enabled': enabled,
            'ratio': ratio,
            'num_pairs': num_pairs,
            'style_a': style_a,
            'style_b': style_b,
            'method': method,
            'allow_empty': allow_empty,
            'min_threshold': min_threshold,
            'batch_size': batch_size,
            'seed': seed,
        }

    def _estimate_style_distance_latent_l1(self, *, num_pairs: int, style_a: int, style_b: int, seed: int) -> float:
        """估计数据集中两个风格（类别）之间的平均差异 D（latent L1）。"""
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data'].get('num_classes'))
        if len(ds.classes) <= max(style_a, style_b):
            raise ValueError(f"[Monitor] 数据集类别数不足：len(classes)={len(ds.classes)}，但需要 style {style_a}/{style_b}")
        if len(ds.files_by_class[style_a]) == 0 or len(ds.files_by_class[style_b]) == 0:
            raise ValueError(f"[Monitor] style_a/style_b 对应的样本为空：{style_a} or {style_b}")

        rng = random.Random(seed)
        diffs: List[float] = []
        for _ in range(max(1, num_pairs)):
            a_path = rng.choice(ds.files_by_class[style_a])
            b_path = rng.choice(ds.files_by_class[style_b])
            a = ds.load_latent(a_path)
            b = ds.load_latent(b_path)
            diffs.append(torch.mean(torch.abs(a - b)).item())
        return float(np.mean(diffs))

    @torch.no_grad()
    def _estimate_style_distance_image(self, *,
                                      num_pairs: int,
                                      style_a: int,
                                      style_b: int,
                                      seed: int,
                                      method: str,
                                      batch_size: int,
                                      latent_dtype: torch.dtype,
                                      decode_latents,
                                      lpips_fn=None) -> float:
        """估计数据集中两个风格（类别）之间的平均差异 D（解码到图像后）。

        method:
          - 'lpips': 需要 lpips_fn
          - 'pixel_l1': 图像像素 L1（在 VAE 输出 [-1,1] 空间）
        """
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data'].get('num_classes'))
        if len(ds.classes) <= max(style_a, style_b):
            raise ValueError(f"[Monitor] 数据集类别数不足：len(classes)={len(ds.classes)}，但需要 style {style_a}/{style_b}")
        if len(ds.files_by_class[style_a]) == 0 or len(ds.files_by_class[style_b]) == 0:
            raise ValueError(f"[Monitor] style_a/style_b 对应的样本为空：{style_a} or {style_b}")

        rng = random.Random(seed)
        method = method.lower().strip()
        if method == 'lpips' and lpips_fn is None:
            raise ValueError("[Monitor] method=lpips 但未提供 lpips_fn")

        diffs: List[float] = []
        pairs = max(1, num_pairs)
        done = 0
        while done < pairs:
            cur_bs = min(max(1, batch_size), pairs - done)
            a_lat = []
            b_lat = []
            for _j in range(cur_bs):
                a_path = rng.choice(ds.files_by_class[style_a])
                b_path = rng.choice(ds.files_by_class[style_b])
                a_lat.append(ds.load_latent(a_path))
                b_lat.append(ds.load_latent(b_path))

            a_b = torch.stack(a_lat, dim=0).to(self.device, dtype=latent_dtype)
            b_b = torch.stack(b_lat, dim=0).to(self.device, dtype=latent_dtype)
            a_img = decode_latents(a_b)
            b_img = decode_latents(b_b)

            if method == 'lpips':
                d = lpips_fn(a_img, b_img).view(-1)
                diffs.extend(d.detach().float().cpu().tolist())
            elif method == 'pixel_l1':
                d = torch.mean(torch.abs(a_img - b_img), dim=[1, 2, 3])
                diffs.extend(d.detach().float().cpu().tolist())
            else:
                raise ValueError(f"[Monitor] 未知 method: {method}")

            done += cur_bs

        return float(np.mean(np.array(diffs, dtype=np.float32)))

    def _base_pair_files(self) -> List[Path]:
        if not self.reflow_dir.exists():
            return []
        return sorted(list(self.reflow_dir.glob("pair_*.pt")))

    def _marker_path(self, name: str) -> Path:
        return self.reflow_dir / name

    def _filter_dir(self, tag: str) -> Path:
        # tag in {"wo", "w"}
        return self.reflow_dir / ("filter_wo_decode" if tag == "wo" else "filter_w_decode")

    def _filter_done_marker(self, tag: str) -> Path:
        return self._filter_dir(tag) / "FILTER_DONE"

    def expected_pair_count(self) -> int:
        # 每个样本会生成 (num_classes - 1) 个目标风格
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data'].get('num_classes'))
        num_classes = len(ds.classes)
        return len(ds) * max(0, num_classes - 1)

    def wait_for_generation_complete(self) -> None:
        """通过文件夹检测等待 run_generation 完成。

        同时使用：
        - 方法 A：达到理论期望 pair 数
        - 方法 B：pair 数在 stable_seconds 内不再增长
        - 增强：检测 GEN_DONE marker
        """
        monitor = self.cfg.get('training', {}).get('generation_monitor', {})
        poll_seconds = float(monitor.get('poll_seconds', 2))
        stable_seconds = float(monitor.get('stable_seconds', 30))
        use_expected = bool(monitor.get('use_expected_count', True))

        expected = self.expected_pair_count() if use_expected else None
        marker = self._marker_path("GEN_DONE")

        print("🕒 [Monitor] 等待 Reflow 数据生成完成...")
        last_count = -1
        stable_start: Optional[float] = None
        while True:
            if marker.exists():
                # marker 存在时也检查一下文件数是否为 0（避免误触发）
                cnt = len(self._base_pair_files())
                if cnt > 0:
                    print(f"✅ [Monitor] 检测到 GEN_DONE，当前 pair 数: {cnt}")
                    return

            cnt = len(self._base_pair_files())
            if expected is not None and cnt >= expected:
                print(f"✅ [Monitor] 已达到期望 pair 数: {cnt}/{expected}")
                return

            now = time.time()
            if cnt != last_count:
                last_count = cnt
                stable_start = now
                if expected is not None:
                    print(f"   ...pair 数变化: {cnt}/{expected}")
                else:
                    print(f"   ...pair 数变化: {cnt}")
            else:
                if stable_start is None:
                    stable_start = now
                if (now - stable_start) >= stable_seconds and cnt > 0:
                    print(f"✅ [Monitor] pair 数在 {stable_seconds}s 内稳定不变，认为生成完成: {cnt}")
                    return

            time.sleep(poll_seconds)

    @staticmethod
    def calc_latent_structure_loss(xc: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """基于 AdaIN 理论的 latent 结构差异：去均值/方差后做 MSE。返回 shape [B]。"""
        eps = 1e-5
        mu_c = torch.mean(xc, dim=[2, 3], keepdim=True)
        std_c = torch.std(xc, dim=[2, 3], keepdim=True) + eps
        mu_z = torch.mean(z1, dim=[2, 3], keepdim=True)
        std_z = torch.std(z1, dim=[2, 3], keepdim=True) + eps
        norm_c = (xc - mu_c) / std_c
        norm_z = (z1 - mu_z) / std_z
        return torch.mean((norm_c - norm_z) ** 2, dim=[1, 2, 3])

    # TODO
    def run_filter_wo_decode(self) -> None:
        """不 decode：Two-Pass 全量扫描 + 百分位筛选，输出到 reflow_dir/filter_wo_decode。"""
        cfg = self.cfg.get('training', {}).get('filter_wo_decode', {})
        if 'keep_percentile' not in cfg:
            raise KeyError("config.json 缺少 training.filter_wo_decode.keep_percentile")
        keep_percentile = float(cfg['keep_percentile'])
        safety_keep_ratio = float(cfg.get('safety_keep_ratio', 0.5))

        out_dir = self._filter_dir('wo')
        out_dir.mkdir(parents=True, exist_ok=True)
        done_marker = self._filter_done_marker('wo')
        if done_marker.exists() and any(out_dir.glob("pair_*.pt")):
            print("✅ [Filter-wo] 已检测到筛选完成标记，跳过。")
            return

        files = self._base_pair_files()
        if not files:
            raise RuntimeError("[Filter-wo] 未找到 pair_*.pt，请先完成 run_generation")

        # 打乱顺序，避免 warmup/筛选偏向某一类顺序
        random.shuffle(files)

        print(f"\n🧪 [Filter-wo] 开始筛选（不 decode）。Two-Pass | 目标保留百分位={keep_percentile}")
        stats_path = out_dir / "filter_stats.jsonl"

        # 监工阈值：如果 z 太接近 content，直接丢弃（防止“偷懒输出原图”）
        mon = self._monitor_cfg(cfg, default_method='latent_l1')
        lazy_threshold: Optional[float] = None
        if mon['enabled']:
            D = self._estimate_style_distance_latent_l1(
                num_pairs=mon['num_pairs'],
                style_a=mon['style_a'],
                style_b=mon['style_b'],
                seed=mon['seed'],
            )
            lazy_threshold = max(mon['min_threshold'], mon['ratio'] * D)
            print(f"🧑‍🏫 [Monitor-wo] D(latent_l1)={D:.6f} | ratio={mon['ratio']} => lazy_threshold={lazy_threshold:.6f}")

        losses: List[Tuple[Path, float, float]] = []  # (file, structure_loss, lazy_metric)

        # Pass 1: 计算所有 loss
        for f in tqdm(files, desc="Filtering-wo (Pass1)"):
            data = torch.load(f, map_location='cpu')
            xc = data['content']
            z1 = data['z']
            if xc.ndim > 3: xc = xc.view(-1, *xc.shape[-3:])[0]
            if z1.ndim > 3: z1 = z1.view(-1, *z1.shape[-3:])[0]
            xc = xc.unsqueeze(0)
            z1 = z1.unsqueeze(0)
            loss = float(self.calc_latent_structure_loss(xc, z1)[0].item())
            lazy_metric = float(torch.mean(torch.abs(z1 - xc)).item())
            losses.append((f, loss, lazy_metric))

        if not losses:
            raise RuntimeError("[Filter-wo] 计算 loss 结果为空")

        # 计算百分位阈值（越小越好）
        loss_values = np.array([v for _, v, _lm in losses], dtype=np.float32)
        threshold = float(np.percentile(loss_values, keep_percentile))

        # Pass 2: 按阈值复制
        kept = 0
        total = len(losses)
        rejected_lazy = 0
        for f, loss, lazy_metric in tqdm(losses, desc="Filtering-wo (Pass2)"):
            out_f = out_dir / f.name
            if out_f.exists():
                kept += 1
                continue
            keep = loss <= threshold
            if lazy_threshold is not None and lazy_metric < lazy_threshold:
                keep = False
                rejected_lazy += 1
            with open(stats_path, 'a', encoding='utf-8') as wf:
                wf.write(json.dumps({
                    "file": f.name,
                    "loss": float(loss),
                    "lazy_metric": float(lazy_metric),
                    "keep": bool(keep),
                    "threshold": threshold,
                    "lazy_threshold": lazy_threshold,
                }, ensure_ascii=False) + "\n")
            if keep:
                shutil.copy2(f, out_f)
                kept += 1

        # Safety Net：如果筛到 0，仅在“没偷懒”的前提下保底（但默认不允许绕过监工）
        if kept == 0:
            candidates = losses
            if lazy_threshold is not None:
                candidates = [x for x in losses if x[2] >= lazy_threshold]
            if not candidates:
                msg = f"[Filter-wo] 监工过滤后无任何样本可用（rejected_lazy={rejected_lazy}/{total}）。请调低 monitor.ratio 或 monitor.min_threshold。"
                if mon.get('allow_empty', False):
                    print("⚠️ " + msg)
                else:
                    raise RuntimeError(msg)
            else:
                k = max(1, int(len(candidates) * safety_keep_ratio))
                topk = sorted(candidates, key=lambda x: x[1])[:k]
                for f, _loss, _lm in topk:
                    out_f = out_dir / f.name
                    if not out_f.exists():
                        shutil.copy2(f, out_f)
                kept = len(list(out_dir.glob('pair_*.pt')))
                with open(stats_path, 'a', encoding='utf-8') as wf:
                    wf.write(json.dumps({"safety_keep_ratio": safety_keep_ratio, "forced_keep": kept, "lazy_threshold": lazy_threshold}, ensure_ascii=False) + "\n")

        done_marker.write_text(
            json.dumps({
                "percentile": keep_percentile,
                "threshold": threshold,
                "kept": kept,
                "total": total,
                "safety_keep_ratio": safety_keep_ratio,
                "lazy_threshold": lazy_threshold,
                "rejected_lazy": rejected_lazy,
                "monitor": mon,
            }, ensure_ascii=False),
            encoding='utf-8'
        )
        print(f"✅ [Filter-wo] 完成：保留 {kept}/{total}（lazy 丢弃 {rejected_lazy}）。阈值={threshold:.6f} 输出目录: {out_dir}")

    # TODO
    @torch.no_grad()
    def run_filter_w_decode(self) -> None:
        """decode + LPIPS：Warm-up 校准阈值，输出到 reflow_dir/filter_w_decode。"""
        cfg = self.cfg.get('training', {}).get('filter_w_decode', {})
        if 'warmup_count' not in cfg:
            raise KeyError("config.json 缺少 training.filter_w_decode.warmup_count")
        if 'warmup_percentile' not in cfg:
            raise KeyError("config.json 缺少 training.filter_w_decode.warmup_percentile")
        warmup_count = int(cfg['warmup_count'])
        warmup_percentile = float(cfg['warmup_percentile'])
        safety_keep_ratio = float(cfg.get('safety_keep_ratio', 0.5))
        fallback_threshold = float(cfg.get('fallback_threshold', 0.5))
        batch_size = int(cfg.get('batch_size', 2))
        use_fp16 = bool(cfg.get('use_fp16', True))
        use_vae_slicing = bool(cfg.get('use_vae_slicing', True))
        use_vae_tiling = bool(cfg.get('use_vae_tiling', True))
        use_checkpointing = bool(cfg.get('use_checkpointing', True))
        decode_chunk_size = int(cfg.get('decode_chunk_size', 1))
        lpips_net = str(cfg.get('lpips_net', 'vgg'))

        try:
            import lpips  # type: ignore
        except ImportError as e:
            raise ImportError("需要安装 lpips 才能使用 filter_w_decode。请先执行: pip install lpips") from e

        out_dir = self._filter_dir('w')
        out_dir.mkdir(parents=True, exist_ok=True)
        done_marker = self._filter_done_marker('w')
        if done_marker.exists() and any(out_dir.glob("pair_*.pt")):
            print("✅ [Filter-w] 已检测到筛选完成标记，跳过。")
            return

        files = self._base_pair_files()
        if not files:
            raise RuntimeError("[Filter-w] 未找到 pair_*.pt，请先完成 run_generation")

        # 打乱顺序，避免 warmup 只覆盖某一类分布
        random.shuffle(files)

        print(f"\n🧪 [Filter-w] 开始筛选（decode + LPIPS）。Warm-up={warmup_count} | percentile={warmup_percentile} | bs={batch_size} | fp16={use_fp16}")

        # 单独加载一份 VAE，避免影响训练/推理时的 self.vae 精度设置
        vae_dtype = torch.float16 if (use_fp16 and self.device.type == 'cuda') else torch.float32
        vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device, dtype=vae_dtype)
        vae.eval(); vae.requires_grad_(False)
        if use_vae_slicing:
            vae.enable_slicing()
        if use_vae_tiling:
            vae.enable_tiling()

        loss_fn = lpips.LPIPS(net=lpips_net).to(self.device)
        loss_fn.eval()
        if vae_dtype == torch.float16:
            loss_fn = loss_fn.half()

        # 监工阈值：如果生成结果和输入过于相似（LPIPS / pixel diff 太小），直接丢弃
        mon = self._monitor_cfg(cfg, default_method='lpips')
        lazy_threshold: Optional[float] = None
        if mon['enabled']:
            # 复用当前 VAE/精度设置，保持阈值与筛选一致
            autocast_ctx = torch.cuda.amp.autocast(enabled=(vae_dtype == torch.float16))
            with autocast_ctx:
                D = self._estimate_style_distance_image(
                    num_pairs=mon['num_pairs'],
                    style_a=mon['style_a'],
                    style_b=mon['style_b'],
                    seed=mon['seed'],
                    method=mon['method'],
                    batch_size=mon['batch_size'],
                    latent_dtype=vae_dtype,
                    decode_latents=decode_latents,
                    lpips_fn=loss_fn if mon['method'].lower().strip() == 'lpips' else None,
                )
            lazy_threshold = max(mon['min_threshold'], mon['ratio'] * D)
            print(f"🧑‍🏫 [Monitor-w] D({mon['method']})={D:.6f} | ratio={mon['ratio']} => lazy_threshold={lazy_threshold:.6f}")

        stats_path = out_dir / "filter_stats.jsonl"
        kept = 0
        total = len(files)
        warmup_losses: List[float] = []
        all_results: List[Tuple[Path, float]] = []
        lpips_threshold: Optional[float] = None
        rejected_lazy = 0

        def decode_latents(latents: torch.Tensor) -> torch.Tensor:
            # latents: [B,4,64,64]
            if use_checkpointing and latents.size(0) > decode_chunk_size:
                outs = []
                for i in range(0, latents.size(0), decode_chunk_size):
                    chunk = latents[i:i + decode_chunk_size]
                    outs.append(vae.decode(chunk / 0.18215).sample)
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                return torch.cat(outs, dim=0)
            return vae.decode(latents / 0.18215).sample  # [-1, 1]

        # 过滤已存在输出（支持断点继续）
        existing = len(list(out_dir.glob("pair_*.pt")))
        kept += existing

        idx = 0
        pbar = tqdm(total=len(files), desc="Filtering-w")
        while idx < len(files):
            chunk = files[idx: idx + batch_size]
            idx += len(chunk)
            # 跳过已存在输出的，减少重复
            pending: List[Path] = []
            for f in chunk:
                out_f = out_dir / f.name
                if out_f.exists():
                    kept += 1
                    pbar.update(1)
                else:
                    pending.append(f)

            if not pending:
                continue

            # 组 batch
            xc_list = []
            z_list = []
            metas = []
            for f in pending:
                data = torch.load(f, map_location='cpu')
                xc = data['content']
                z1 = data['z']
                if xc.ndim > 3: xc = xc.view(-1, *xc.shape[-3:])[0]
                if z1.ndim > 3: z1 = z1.view(-1, *z1.shape[-3:])[0]
                xc_list.append(xc)
                z_list.append(z1)
                metas.append(f)

            xc_b = torch.stack(xc_list, dim=0).to(self.device, dtype=vae_dtype, non_blocking=True)
            z_b = torch.stack(z_list, dim=0).to(self.device, dtype=vae_dtype, non_blocking=True)

            try:
                autocast_ctx = torch.cuda.amp.autocast(enabled=(vae_dtype == torch.float16))
                with autocast_ctx:
                    xc_img = decode_latents(xc_b)
                    z_img = decode_latents(z_b)
                    # LPIPS 输入要求 [-1,1]，这里正好符合
                    dists = loss_fn(xc_img, z_img).view(-1)

                for f, dist in zip(metas, dists.detach().float().cpu().tolist()):
                    dist_f = float(dist)
                    all_results.append((f, dist_f))

                    # Warm-up 先积累，不做筛
                    if len(warmup_losses) < warmup_count:
                        warmup_losses.append(dist_f)
                        pbar.update(1)
                        continue

                    # 计算阈值（仅第一次）
                    if lpips_threshold is None:
                        if len(warmup_losses) > 0:
                            lpips_threshold = float(np.percentile(np.array(warmup_losses, dtype=np.float32), warmup_percentile))
                        else:
                            lpips_threshold = fallback_threshold
                        print(f"\n📌 [Filter-w] Warm-up 阈值校准完成：{lpips_threshold:.6f}")

                    keep = dist_f <= lpips_threshold
                    if lazy_threshold is not None and dist_f < lazy_threshold:
                        keep = False
                        rejected_lazy += 1
                    with open(stats_path, 'a', encoding='utf-8') as wf:
                        wf.write(json.dumps({
                            "file": f.name,
                            "metric": mon['method'],
                            "value": dist_f,
                            "keep": bool(keep),
                            "threshold": lpips_threshold,
                            "lazy_threshold": lazy_threshold,
                        }, ensure_ascii=False) + "\n")
                    if keep:
                        shutil.copy2(f, out_dir / f.name)
                        kept += 1
                    pbar.update(1)

            except RuntimeError as e:
                # 显存 OOM 时退化为单张处理，尽量不中断整晚运行
                if "out of memory" not in str(e).lower():
                    raise
                print("\n⚠️ [Filter-w] 检测到 OOM，自动降级为单张处理并清理缓存...")
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                # 退化逐个处理
                for f in metas:
                    try:
                        data = torch.load(f, map_location='cpu')
                        xc = data['content']
                        z1 = data['z']
                        if xc.ndim > 3: xc = xc.view(-1, *xc.shape[-3:])[0]
                        if z1.ndim > 3: z1 = z1.view(-1, *z1.shape[-3:])[0]
                        xc_b = xc.unsqueeze(0).to(self.device, dtype=vae_dtype)
                        z_b = z1.unsqueeze(0).to(self.device, dtype=vae_dtype)
                        autocast_ctx = torch.cuda.amp.autocast(enabled=(vae_dtype == torch.float16))
                        with autocast_ctx:
                            xc_img = decode_latents(xc_b)
                            z_img = decode_latents(z_b)
                            dist = loss_fn(xc_img, z_img).view(-1)[0].detach().float().cpu().item()
                        dist_f = float(dist)
                        all_results.append((f, dist_f))

                        if len(warmup_losses) < warmup_count:
                            warmup_losses.append(dist_f)
                            pbar.update(1)
                            continue

                        if lpips_threshold is None:
                            if len(warmup_losses) > 0:
                                lpips_threshold = float(np.percentile(np.array(warmup_losses, dtype=np.float32), warmup_percentile))
                            else:
                                lpips_threshold = fallback_threshold
                            print(f"\n📌 [Filter-w] Warm-up 阈值校准完成：{lpips_threshold:.6f}")

                        keep = dist_f <= lpips_threshold
                        if lazy_threshold is not None and dist_f < lazy_threshold:
                            keep = False
                            rejected_lazy += 1
                        with open(stats_path, 'a', encoding='utf-8') as wf:
                            wf.write(json.dumps({
                                "file": f.name,
                                "metric": mon['method'],
                                "value": dist_f,
                                "keep": bool(keep),
                                "threshold": lpips_threshold,
                                "lazy_threshold": lazy_threshold,
                            }, ensure_ascii=False) + "\n")
                        if keep:
                            shutil.copy2(f, out_dir / f.name)
                            kept += 1
                        pbar.update(1)
                    except Exception as ee:
                        # 单样本失败也不中断；记录并继续
                        with open(stats_path, 'a', encoding='utf-8') as wf:
                            wf.write(json.dumps({"file": f.name, "error": str(ee)}, ensure_ascii=False) + "\n")
                        pbar.update(1)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        pbar.close()

        # 如果 warmup 阈值还未生成（数据量不足），这里补一次
        if lpips_threshold is None:
            if len(warmup_losses) > 0:
                lpips_threshold = float(np.percentile(np.array(warmup_losses, dtype=np.float32), warmup_percentile))
            else:
                lpips_threshold = fallback_threshold
            print(f"\n📌 [Filter-w] Warm-up 阈值校准完成：{lpips_threshold:.6f}")

        # Warm-up 阶段积累的样本，需要在这里补筛
        if warmup_losses:
            warmup_count_actual = min(warmup_count, len(all_results))
            for f, dist_f in all_results[:warmup_count_actual]:
                keep = dist_f <= lpips_threshold
                if lazy_threshold is not None and dist_f < lazy_threshold:
                    keep = False
                    rejected_lazy += 1
                with open(stats_path, 'a', encoding='utf-8') as wf:
                    wf.write(json.dumps({
                        "file": f.name,
                        "metric": mon['method'],
                        "value": dist_f,
                        "keep": bool(keep),
                        "threshold": lpips_threshold,
                        "lazy_threshold": lazy_threshold,
                        "warmup": True,
                    }, ensure_ascii=False) + "\n")
                if keep:
                    out_f = out_dir / f.name
                    if not out_f.exists():
                        shutil.copy2(f, out_f)
                        kept += 1

        # Safety Net：如果筛到 0，默认不绕过“监工”，仅在允许时做保底
        if kept == 0 and all_results:
            candidates = all_results
            if lazy_threshold is not None:
                candidates = [x for x in all_results if x[1] >= lazy_threshold]
            if not candidates:
                msg = f"[Filter-w] 监工过滤后无任何样本可用（rejected_lazy={rejected_lazy}/{len(all_results)}）。请调低 monitor.ratio 或改用 pixel_l1。"
                if mon.get('allow_empty', False):
                    print("⚠️ " + msg)
                else:
                    raise RuntimeError(msg)
            else:
                # 尽量优先选落在 [lazy_threshold, lpips_threshold] 区间内的
                in_band = candidates
                if lpips_threshold is not None:
                    in_band = [x for x in candidates if x[1] <= lpips_threshold]
                pool = in_band if in_band else candidates
                k = max(1, int(len(pool) * safety_keep_ratio))
                topk = sorted(pool, key=lambda x: x[1])[:k]
                for f, _ in topk:
                    out_f = out_dir / f.name
                    if not out_f.exists():
                        shutil.copy2(f, out_f)
                kept = len(list(out_dir.glob('pair_*.pt')))
                with open(stats_path, 'a', encoding='utf-8') as wf:
                    wf.write(json.dumps({"safety_keep_ratio": safety_keep_ratio, "forced_keep": kept, "lazy_threshold": lazy_threshold}, ensure_ascii=False) + "\n")

        done_marker.write_text(
            json.dumps({
                "threshold": lpips_threshold,
                "metric": mon['method'],
                "kept": kept,
                "total": total,
                "fp16": bool(use_fp16),
                "warmup_count": warmup_count,
                "warmup_percentile": warmup_percentile,
                "safety_keep_ratio": safety_keep_ratio,
                "lazy_threshold": lazy_threshold,
                "rejected_lazy": rejected_lazy,
                "monitor": mon,
            }, ensure_ascii=False),
            encoding='utf-8'
        )
        print(f"✅ [Filter-w] 完成：保留 {kept}/{total}（lazy 丢弃 {rejected_lazy}）。阈值={lpips_threshold:.6f} 输出目录: {out_dir}")

    def get_model(self):
        return SAFModel(**self.cfg['model']).to(self.device)

    def resume_checkpoint(self, model, stage_prefix):
        ckpts = list(self.ckpt_dir.glob(f"{stage_prefix}_epoch*.pt"))
        if not ckpts: return 1
        latest = max(ckpts, key=lambda p: int(re.search(r'epoch(\d+)', p.name).group(1)))
        model.load_state_dict(torch.load(latest, map_location=self.device))
        epoch = int(re.search(r'epoch(\d+)', latest.name).group(1))
        print(f"🟢 [Resume] 已从 Epoch {epoch} 恢复权重")
        return epoch + 1

    def make_balanced_sampler(self, dataset):
        print("⚖️ [Sampler] 正在计算类别均衡权重...")
        targets = [cls_id for _, cls_id in dataset.all_files]
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts
        sample_weights = np.array([class_weights[t] for t in targets])
        
        return WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights), 
            num_samples=len(sample_weights), 
            replacement=True
        )

    def make_stage2_sampler(self, dataset: Stage2Dataset) -> Optional[WeightedRandomSampler]:
        """按 src_label 做 Stage2 均衡采样。

        目标：实现 1:1 采样（如真实:莫奈）。样本较少的类会在一个 epoch 内被重复取样。
        """
        cfg = dict(self.cfg.get('training', {}).get('stage2_sampler', {}) or {})
        enabled = bool(cfg.get('enabled', True))
        if not enabled:
            return None

        if not hasattr(dataset, 'src_labels') or not dataset.src_labels:
            print("⚠️ [Sampler] Stage2 未提供 src_labels，跳过均衡采样。")
            return None

        labels = list(dataset.src_labels)
        # 过滤无效标签
        valid_idx = [i for i, v in enumerate(labels) if isinstance(v, int) and v >= 0]
        if not valid_idx:
            print("⚠️ [Sampler] Stage2 src_labels 全部无效，跳过均衡采样。")
            return None

        counts: Dict[int, int] = {}
        for i in valid_idx:
            v = labels[i]
            counts[v] = counts.get(v, 0) + 1

        num_classes = len(counts)
        if num_classes <= 1:
            print("⚠️ [Sampler] Stage2 仅检测到 1 个 src 类别，跳过均衡采样。")
            return None

        max_count = max(counts.values())
        # 令每个类别在一个 epoch 内期望出现 max_count 次 -> 1:1:... 均衡
        num_samples = max_count * num_classes

        weights = []
        for i in range(len(labels)):
            v = labels[i]
            if v in counts and counts[v] > 0:
                weights.append(1.0 / counts[v])
            else:
                weights.append(0.0)

        print(f"⚖️ [Sampler] Stage2 启用 src 均衡采样：classes={counts} | num_samples={num_samples}")
        return WeightedRandomSampler(
            weights=torch.tensor(weights, dtype=torch.double),
            num_samples=int(num_samples),
            replacement=True
        )
    
    # TODO
    @torch.no_grad()
    def do_inference(self, model, dataset, epoch, stage):
        model.eval()

        # 仅在可视化阶段把 VAE 临时搬到 GPU（如可用），画完再放回 CPU
        if self.device.type == 'cuda':
            if self.keep_vae_on_gpu or (next(self.vae.parameters()).device.type != 'cuda'):
                self.vae.to(self.device)

        # CFG（Classifier-Free Guidance）：使用 null_style 作为 uncond
        cfg_scale = float(
            self.cfg.get('inference', {}).get(
                'cfg_scale',
                self.cfg.get('training', {}).get('cfg_scale', 1.0),
            )
        )
        null_style = torch.tensor([self.null_style_id], device=self.device, dtype=torch.long)
        save_dir = self.vis_root / stage / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        num_classes = len(dataset.classes)
        print(f"🖼️ [Inference] 采样推理中...")
        
        for src_id in range(num_classes):
            available_files = dataset.files_by_class[src_id]
            selected_files = random.sample(available_files, min(NUM_SAMPLES_PER_CLASS, len(available_files)))
            
            for f_idx, sample_path in enumerate(selected_files):
                x_c = dataset.load_latent(sample_path).unsqueeze(0).to(self.device)
                
                # 原图参考
                orig = self.vae.decode(x_c / 0.18215).sample
                orig = (orig / 2 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()[0]
                Image.fromarray((orig * 255).astype('uint8')).save(save_dir / f"src_cls{src_id}_samp{f_idx}_orig.jpg")
                
                for target_id in range(num_classes):
                    x_t = x_c.clone()
                    tid_tensor = torch.tensor([target_id], device=self.device)
                    for i in range(20):
                        t = torch.ones(1, device=self.device) * (i / 20)

                        # --- CFG 核心逻辑 ---
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
                        # -------------------

                        x_t = x_t + v * (1/20)
                    
                    res = self.vae.decode(x_t / 0.18215).sample
                    res = (res / 2 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()[0]
                    
                    suffix = "_ID" if src_id == target_id else ""
                    Image.fromarray((res * 255).astype('uint8')).save(
                        save_dir / f"src{src_id}_to{target_id}{suffix}.jpg"
                    )

        # 可视化结束：释放 VAE 的显存占用
        if self.device.type == 'cuda' and (not self.keep_vae_on_gpu):
            self.vae.to('cpu')
            torch.cuda.empty_cache()
        model.train()

    def run_stage1(self):
        print("\n🚀 [Stage 1] 开始均衡化训练...")
        model = self.get_model()
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])

        ema = None
        if self.ema_enabled:
            print("🛡️ [Init] 初始化 EMA...")
            ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_decay)
            ema.to(self.device)
        
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data']['num_classes'])
        sampler = self.make_balanced_sampler(ds)

        num_workers = int(self.cfg['training'].get('num_workers', 0))
        
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], 
                sampler=sampler, shuffle=False, drop_last=True,
                num_workers=num_workers,
                pin_memory=(self.device.type == 'cuda'),
                persistent_workers=(num_workers > 0))
        
        start_epoch = self.resume_checkpoint(model, "stage1")

        # ===== 学习率调度（Warmup + Cosine Decay） =====
        if get_cosine_schedule_with_warmup is None:
            raise ImportError("需要安装 transformers 才能使用 get_cosine_schedule_with_warmup。请先执行: pip install transformers")

        stage1_warmup_ratio = float(self.cfg.get('training', {}).get('stage1_warmup_ratio', 0.1))
        total_steps = max(1, len(dl) * int(self.cfg['training']['stage1_epochs']))
        warmup_steps = int(total_steps * max(0.0, stage1_warmup_ratio))
        completed_steps = max(0, (start_epoch - 1) * len(dl))
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            last_epoch=completed_steps - 1,
        )

        # TensorBoard step（尽量和 epoch 对齐，resume 时保持连续）
        tb_step = (start_epoch - 1) * len(dl)
        
        for epoch in range(start_epoch, self.cfg['training']['stage1_epochs'] + 1):
            model.train()
            pbar = tqdm(
                dl,
                desc=f"S1 Ep {epoch}",
                dynamic_ncols=self.tqdm_dynamic_ncols,
                ncols=self.tqdm_ncols,
            )
            history = {'Lc': [], 'Ls': []}
            it = 0
            checked_data_this_epoch = False
            
            for x_c, x_s, t_id, s_id in pbar:
                x_c, x_s, t_id, s_id = x_c.to(self.device), x_s.to(self.device), t_id.to(self.device), s_id.to(self.device)
                B = x_c.size(0)

                # --- 🚨 数据异常检测与可选修复（每个 epoch 仅检查一次） ---
                if self.data_check_enabled and (not checked_data_this_epoch):
                    checked_data_this_epoch = True
                    with torch.no_grad():
                        absmean = x_c.detach().float().abs().mean()
                        vmax = x_c.detach().float().amax()
                        is_suspicious = (absmean > self.data_check_absmean_threshold) or (vmax > self.data_check_max_threshold)
                        if bool(is_suspicious.detach().cpu().item()):
                            if (not self.data_check_print_once) or (not self._data_alert_printed):
                                self._data_alert_printed = True
                                try:
                                    print("\n⚠️⚠️⚠️ [Data Alert] 检测到输入张量数值异常（疑似未归一化像素 0-255 被当作 latent）")
                                    print(f"    x_c shape={tuple(x_c.shape)} dtype={x_c.dtype} device={x_c.device}")
                                    print(f"    x_c absmean={float(absmean.detach().cpu().item()):.2f}, mean={float(x_c.detach().float().mean().cpu().item()):.2f}, max={float(vmax.detach().cpu().item()):.2f}")
                                    print(f"    建议先确认 Stage1Dataset 返回的是 SD latent（通常是 4x64x64 且数值在 ~[-5,5] 左右）")
                                    if self.data_check_auto_fix:
                                        print(f"    将按 fix_mode={self.data_check_fix_mode} 做临时缩放（仅用于排查，不建议长期依赖）")
                                except Exception:
                                    pass

                            if self.data_check_auto_fix:
                                if self.data_check_fix_mode == '0_255_to_minus1_1':
                                    x_c = (x_c / 255.0) * 2.0 - 1.0
                                    x_s = (x_s / 255.0) * 2.0 - 1.0
                                elif self.data_check_fix_mode == '0_255_to_0_1':
                                    x_c = x_c / 255.0
                                    x_s = x_s / 255.0
                                else:
                                    raise ValueError(f"未知 data_sanity_check.fix_mode: {self.data_check_fix_mode}")
                # --------------------------------
                
                is_self = torch.rand(B, device=self.device) < IDENTITY_RATE
                x_target = torch.where(is_self.view(-1,1,1,1), x_c, x_s)
                style_id = torch.where(is_self, s_id, t_id)
                style_id = self._apply_style_dropout(style_id)

                opt.zero_grad(set_to_none=True)
                t = torch.rand(B, device=self.device)
                x_t = (1 - t.view(-1, 1, 1, 1)) * x_c + t.view(-1, 1, 1, 1) * x_target

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    v_pred = model(x_t, x_c, t, style_id)
                    v_target = x_target - x_c

                    # 计算逐样本 MSE
                    loss_elementwise = torch.mean((v_pred - v_target) ** 2, dim=[1, 2, 3])

                    # 🔴 应用类别特定的 Identity 权重
                    weights = torch.ones(B, device=self.device)
                    if is_self.any():
                        # 仅在 identity 样本上应用 config 中的权重
                        weights[is_self] = self.id_loss_weights[s_id[is_self]]

                    loss_weighted = loss_elementwise * weights

                    # Gram style loss（只在 style 变换样本上计算，默认不作用于 identity）
                    if self.gram_enabled and self.gram_lambda > 0:
                        # 从 v_pred 构造预测的终点（近似 x_out ≈ x_c + v_pred）
                        x_out_pred = x_c + v_pred
                        gram_per = self._gram_style_loss(x_out_pred, x_target)

                        if self.gram_apply_on == 'style_only':
                            gram_per = gram_per * (~is_self).float()
                            # IMPORTANT: 不要用 .item()，否则每步强制 GPU 同步
                            gram_denom = (~is_self).float().sum().clamp_min(1.0)
                            gram_loss = gram_per.sum() / gram_denom
                        else:
                            gram_loss = gram_per.mean()

                        loss = loss_weighted.mean() + self.gram_lambda * gram_loss
                    else:
                        gram_loss = torch.zeros((), device=self.device)
                        loss = loss_weighted.mean()

                # 防御：loss 非有限时跳过当前 batch
                if self.grad_skip_on_nonfinite_loss and (not bool(torch.isfinite(loss.detach()).cpu().item())):
                    try:
                        print(f"⚠️ [Warning] Stage1 Epoch {epoch} loss is NaN/Inf, skipping batch.")
                    except Exception:
                        pass
                    opt.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.update()
                    tb_step += 1
                    it += 1
                    continue

                did_step = False
                grad_norm: Optional[torch.Tensor] = None

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)

                    # 梯度裁剪/爆炸检测：若 clip_max_norm<=0，则用 inf 只计算 norm 不裁剪
                    need_norm = (self.grad_clip_max_norm > 0) or (self.grad_skip_norm_threshold > 0)
                    if need_norm:
                        max_norm = self.grad_clip_max_norm if (self.grad_clip_max_norm > 0) else float('inf')
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                    # 梯度爆炸：nonfinite 或超过阈值则跳过该 batch
                    if grad_norm is not None:
                        bad = (not bool(torch.isfinite(grad_norm.detach()).cpu().item()))
                        too_big = (self.grad_skip_norm_threshold > 0) and (float(grad_norm.detach().cpu().item()) > self.grad_skip_norm_threshold)
                        if bad or too_big:
                            try:
                                print(f"⚠️ [Warning] Stage1 Epoch {epoch} grad_norm={'NaN/Inf' if bad else float(grad_norm.detach().cpu().item()):.4f} too large, skipping batch.")
                            except Exception:
                                pass
                            opt.zero_grad(set_to_none=True)
                            self.scaler.update()
                            tb_step += 1
                            it += 1
                            continue

                    self.scaler.step(opt)
                    self.scaler.update()
                    did_step = True
                else:
                    loss.backward()

                    need_norm = (self.grad_clip_max_norm > 0) or (self.grad_skip_norm_threshold > 0)
                    if need_norm:
                        max_norm = self.grad_clip_max_norm if (self.grad_clip_max_norm > 0) else float('inf')
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                        bad = (not bool(torch.isfinite(grad_norm.detach()).cpu().item()))
                        too_big = (self.grad_skip_norm_threshold > 0) and (float(grad_norm.detach().cpu().item()) > self.grad_skip_norm_threshold)
                        if bad or too_big:
                            try:
                                print(f"⚠️ [Warning] Stage1 Epoch {epoch} grad_norm={'NaN/Inf' if bad else float(grad_norm.detach().cpu().item()):.4f} too large, skipping batch.")
                            except Exception:
                                pass
                            opt.zero_grad(set_to_none=True)
                            tb_step += 1
                            it += 1
                            continue

                    opt.step()
                    did_step = True

                if did_step:
                    scheduler.step()
                if did_step and (ema is not None):
                    ema.update()
                
                # ===== 监控（进度条 + TensorBoard） =====
                do_monitor = (it % max(1, self.monitor_every)) == 0
                if do_monitor:
                    with torch.no_grad():
                        has_self = bool(is_self.any().detach().cpu().item())
                        has_trans = bool((~is_self).any().detach().cpu().item())

                        ls_val = float(loss_elementwise[is_self].mean().detach().cpu().item()) if has_self else 0.0
                        lc_val = float(loss_elementwise[~is_self].mean().detach().cpu().item()) if has_trans else 0.0
                        gram_val = float(gram_loss.detach().cpu().item()) if (self.gram_enabled and self.gram_lambda > 0) else 0.0

                        if has_trans:
                            v_tr = v_pred[~is_self].detach()
                            v_mag_transfer = float(v_tr.flatten(1).norm(p=2, dim=1).mean().detach().cpu().item())
                        else:
                            v_mag_transfer = 0.0

                        loss_val = float(loss.detach().cpu().item())

                        grad_norm_val = float(grad_norm.detach().cpu().item()) if grad_norm is not None else 0.0

                    history.setdefault('Gram', []).append(gram_val)
                    history.setdefault('VMag', []).append(v_mag_transfer)
                    history['Ls'].append(ls_val)
                    history['Lc'].append(lc_val)

                    pbar.set_postfix({
                        "Flow(Trans)": f"{np.mean(history['Lc'][-20:]):.4f}",
                        "Flow(ID)":    f"{np.mean(history['Ls'][-20:]):.4f}",
                        "Gram":        f"{np.mean(history['Gram'][-20:]):.4f}",
                        "vMag":        f"{np.mean(history['VMag'][-20:]):.3f}",
                    })

                    if self.tb_writer is not None and ((tb_step % max(1, self.tb_log_every)) == 0):
                        self.tb_writer.add_scalar('stage1/FlowTrans_Lc', lc_val, tb_step)
                        self.tb_writer.add_scalar('stage1/FlowID_Ls', ls_val, tb_step)
                        self.tb_writer.add_scalar('stage1/Gram', gram_val, tb_step)
                        self.tb_writer.add_scalar('stage1/vMag_transfer', v_mag_transfer, tb_step)
                        self.tb_writer.add_scalar('stage1/loss_total', loss_val, tb_step)
                        self.tb_writer.add_scalar('stage1/grad_norm', grad_norm_val, tb_step)
                        if self.use_amp:
                            try:
                                self.tb_writer.add_scalar('stage1/grad_scale', float(self.scaler.get_scale()), tb_step)
                            except Exception:
                                pass
                        self.tb_writer.add_scalar('stage1/lr', float(opt.param_groups[0]['lr']), tb_step)

                tb_step += 1
                it += 1

            # 每个 epoch 打印一次学习率
            try:
                print(f"📈 [Stage1] Epoch {epoch} lr={opt.param_groups[0]['lr']:.6g}")
            except Exception:
                pass

            if self.eval_step > 0 and (epoch % self.eval_step == 0):
                # 保存 raw（用于断点续训）
                torch.save(model.state_dict(), self.ckpt_dir / f"stage1_epoch{epoch}.pt")
                torch.save(model.state_dict(), self.ckpt_dir / f"stage1_epoch{epoch}_raw.pt")

                # 保存 EMA 权重并可选推理
                if ema is not None:
                    with ema.average_parameters():
                        torch.save(model.state_dict(), self.ckpt_dir / f"stage1_epoch{epoch}_ema.pt")
                        if self.do_infer:
                            self.do_inference(model, ds, epoch, "stage1")
                else:
                    if self.do_infer:
                        self.do_inference(model, ds, epoch, "stage1")
                
        # 保存 final：维持 stage1_final.pt 给下游使用（优先 EMA）
        torch.save(model.state_dict(), self.ckpt_dir / "stage1_final_raw.pt")
        if ema is not None:
            with ema.average_parameters():
                torch.save(model.state_dict(), self.ckpt_dir / "stage1_final_ema.pt")
                torch.save(model.state_dict(), self.ckpt_dir / "stage1_final.pt")
        else:
            torch.save(model.state_dict(), self.ckpt_dir / "stage1_final.pt")

    @torch.no_grad()
    def run_generation(self):
        print("\n🌊 [Reflow] 数据合成中...")
        # 重新生成会清空整个缓存目录（包含筛选结果和标记），确保结果一致
        if self.reflow_dir.exists():
            shutil.rmtree(self.reflow_dir)
        self.reflow_dir.mkdir(parents=True, exist_ok=True)
        model = self.get_model()
        model.load_state_dict(torch.load(self.ckpt_dir / "stage1_final.pt", map_location=self.device))
        model.eval()
        ds = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data'].get('num_classes'))
        num_workers = int(self.cfg['training'].get('num_workers', 0))
        dl = DataLoader(ds, batch_size=self.cfg['training']['batch_size'], shuffle=False,
                num_workers=num_workers,
                pin_memory=(self.device.type == 'cuda'),
                persistent_workers=(num_workers > 0))
        cnt = 0
        for x_c, _, _, s_id in tqdm(dl, desc="Synthesizing"):
            x_c, s_id = x_c.to(self.device), s_id.to(self.device)
            for target_id in range(len(ds.classes)):
                t_ids = torch.full((x_c.size(0),), target_id, dtype=torch.long, device=self.device)
                mask = (s_id != t_ids)
                if mask.sum() == 0: continue
                curr_xc, curr_tid, curr_sid = x_c[mask], t_ids[mask], s_id[mask]
                xt = curr_xc.clone()
                for i in range(20):
                    t = torch.ones(curr_xc.size(0), device=self.device) * (i / 20)
                    xt = xt + model(xt, curr_xc, t, curr_tid) * (1/20)
                for i in range(curr_xc.size(0)):
                    torch.save({'content': curr_xc[i].cpu(), 'z': xt[i].cpu(), 'style_label': curr_tid[i].cpu(), 'src_label': curr_sid[i].cpu()}, 
                               self.reflow_dir / f"pair_{cnt}_src{curr_sid[i].item()}.pt")
                    cnt += 1
        # 生成完成 marker（增强可靠性）
        self._marker_path("GEN_DONE").write_text(json.dumps({"count": cnt}, ensure_ascii=False), encoding='utf-8')
        print(f"✅ [Reflow] 生成完成：共 {cnt} 个 pairs")

    def run_stage2(self, data_dir: Path, tag: str):
        """Stage2 训练（按 tag 分开保存 ckpt / 可视化 / final）。

        tag: "wo" or "w"
        data_dir: 对应筛选后的 pair 目录
        """
        stage_name = f"stage2_{tag}"
        ckpt_prefix = f"stage2_{tag}"
        final_name = f"saf_final_reflowed_{tag}.pt"

        print(f"\n✨ [Stage 2-{tag}] 直线拉直训练... 数据目录: {data_dir}")
        model = self.get_model()

        ema = None
        if self.ema_enabled:
            print("🛡️ [Init] 初始化 EMA...")
            ema = ExponentialMovingAverage(model.parameters(), decay=self.ema_decay)
            ema.to(self.device)

        # 两套逻辑独立：各自只 resume 自己的 ckpt
        start_epoch = self.resume_checkpoint(model, ckpt_prefix)
        if start_epoch == 1 and bool(self.cfg.get('training', {}).get('stage2_init_from_stage1', True)):
            # Stage2 初始化使用 Stage1 的 ema 权重
            s1 = self.ckpt_dir / "stage1_final.pt"
            if s1.exists():
                print("🧷 [Stage2] 使用 stage1_final 初始化权重")
                model.load_state_dict(torch.load(s1, map_location=self.device))

        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg['training']['learning_rate'])
        ds = Stage2Dataset(data_dir)
        ds_infer = Stage1Dataset(self.cfg['data']['data_root'], self.cfg['data'].get('num_classes'))
        sampler = self.make_stage2_sampler(ds)
        dl = DataLoader(
            ds,
            batch_size=self.cfg['training']['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=True,
            num_workers=self.cfg['training'].get('num_workers', 0),
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=(int(self.cfg['training'].get('num_workers', 0)) > 0),
        )

        tb_step = (start_epoch - 1) * len(dl)

        # ===== 学习率调度（Stage2 仅 Cosine Decay，无 Warmup） =====
        if get_cosine_schedule_with_warmup is None:
            raise ImportError("需要安装 transformers 才能使用 get_cosine_schedule_with_warmup。请先执行: pip install transformers")

        total_steps = max(1, len(dl) * int(self.cfg['training']['stage2_epochs']))
        warmup_steps = 0
        completed_steps = max(0, (start_epoch - 1) * len(dl))
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            last_epoch=completed_steps - 1,
        )

        for epoch in range(start_epoch, self.cfg['training']['stage2_epochs'] + 1):
            model.train()
            pbar = tqdm(
                dl,
                desc=f"S2-{tag} Ep {epoch}",
                dynamic_ncols=self.tqdm_dynamic_ncols,
                ncols=self.tqdm_ncols,
            )
            history = {'Flow': [], 'Gram': [], 'VMag': []}
            for x_c, z, s_id in pbar:
                x_c, z, s_id = x_c.to(self.device), z.to(self.device), s_id.to(self.device)
                s_id = self._apply_style_dropout(s_id)
                opt.zero_grad(set_to_none=True)
                t = torch.rand(x_c.size(0), device=self.device).view(-1,1,1,1)
                x_t = (1 - t) * x_c + t * z
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    v_pred = model(x_t, x_c, t.squeeze(), s_id)
                    flow_loss = torch.mean((v_pred - (z - x_c))**2)

                if self.gram_enabled and self.gram_lambda > 0:
                    x_out_pred = x_c + v_pred
                    gram_per = self._gram_style_loss(x_out_pred, z)
                    gram_loss = gram_per.mean()
                    loss = flow_loss + self.gram_lambda * gram_loss
                else:
                    gram_loss = torch.zeros((), device=self.device)
                    loss = flow_loss

                # 防御：loss 非有限时跳过当前 batch
                if self.grad_skip_on_nonfinite_loss and (not bool(torch.isfinite(loss.detach()).cpu().item())):
                    try:
                        print(f"⚠️ [Warning] Stage2-{tag} Epoch {epoch} loss is NaN/Inf, skipping batch.")
                    except Exception:
                        pass
                    opt.zero_grad(set_to_none=True)
                    if self.use_amp:
                        self.scaler.update()
                    tb_step += 1
                    continue

                did_step = False
                grad_norm: Optional[torch.Tensor] = None

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(opt)

                    need_norm = (self.grad_clip_max_norm > 0) or (self.grad_skip_norm_threshold > 0)
                    if need_norm:
                        max_norm = self.grad_clip_max_norm if (self.grad_clip_max_norm > 0) else float('inf')
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

                    if grad_norm is not None:
                        bad = (not bool(torch.isfinite(grad_norm.detach()).cpu().item()))
                        too_big = (self.grad_skip_norm_threshold > 0) and (float(grad_norm.detach().cpu().item()) > self.grad_skip_norm_threshold)
                        if bad or too_big:
                            try:
                                print(f"⚠️ [Warning] Stage2-{tag} Epoch {epoch} grad_norm={'NaN/Inf' if bad else float(grad_norm.detach().cpu().item()):.4f} too large, skipping batch.")
                            except Exception:
                                pass
                            opt.zero_grad(set_to_none=True)
                            self.scaler.update()
                            tb_step += 1
                            continue

                    self.scaler.step(opt)
                    self.scaler.update()
                    did_step = True
                else:
                    loss.backward()
                    need_norm = (self.grad_clip_max_norm > 0) or (self.grad_skip_norm_threshold > 0)
                    if need_norm:
                        max_norm = self.grad_clip_max_norm if (self.grad_clip_max_norm > 0) else float('inf')
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
                        bad = (not bool(torch.isfinite(grad_norm.detach()).cpu().item()))
                        too_big = (self.grad_skip_norm_threshold > 0) and (float(grad_norm.detach().cpu().item()) > self.grad_skip_norm_threshold)
                        if bad or too_big:
                            try:
                                print(f"⚠️ [Warning] Stage2-{tag} Epoch {epoch} grad_norm={'NaN/Inf' if bad else float(grad_norm.detach().cpu().item()):.4f} too large, skipping batch.")
                            except Exception:
                                pass
                            opt.zero_grad(set_to_none=True)
                            tb_step += 1
                            continue
                    opt.step()
                    did_step = True

                if did_step:
                    scheduler.step()
                if did_step and (ema is not None):
                    ema.update()

                with torch.no_grad():
                    v_mag = v_pred.detach().view(v_pred.size(0), -1).norm(p=2, dim=1).mean().item()
                    grad_norm_val = float(grad_norm.detach().cpu().item()) if grad_norm is not None else 0.0
                gram_val = gram_loss.item() if (self.gram_enabled and self.gram_lambda > 0) else 0.0
                history['Flow'].append(float(flow_loss.item()))
                history['Gram'].append(float(gram_val))
                history['VMag'].append(float(v_mag))
                pbar.set_postfix({
                    "Flow": f"{np.mean(history['Flow'][-20:]):.4f}",
                    "Gram": f"{np.mean(history['Gram'][-20:]):.4f}",
                    "vMag": f"{np.mean(history['VMag'][-20:]):.3f}",
                })

                if self.tb_writer is not None:
                    if (tb_step % max(1, self.tb_log_every)) == 0:
                        self.tb_writer.add_scalar(f'stage2_{tag}/flow', float(flow_loss.item()), tb_step)
                        self.tb_writer.add_scalar(f'stage2_{tag}/gram', float(gram_val), tb_step)
                        self.tb_writer.add_scalar(f'stage2_{tag}/vMag', float(v_mag), tb_step)
                        self.tb_writer.add_scalar(f'stage2_{tag}/loss_total', float(loss.item()), tb_step)
                        self.tb_writer.add_scalar(f'stage2_{tag}/grad_norm', float(grad_norm_val), tb_step)
                        if self.use_amp:
                            try:
                                self.tb_writer.add_scalar(f'stage2_{tag}/grad_scale', float(self.scaler.get_scale()), tb_step)
                            except Exception:
                                pass
                        self.tb_writer.add_scalar(f'stage2_{tag}/lr', float(opt.param_groups[0]['lr']), tb_step)
                tb_step += 1

            # 每个 epoch 打印一次学习率
            try:
                print(f"📈 [Stage2-{tag}] Epoch {epoch} lr={opt.param_groups[0]['lr']:.6g}")
            except Exception:
                pass

            if epoch % EVAL_STEP == 0:
                # 保存 raw（用于断点续训）
                torch.save(model.state_dict(), self.ckpt_dir / f"{ckpt_prefix}_epoch{epoch}.pt")
                torch.save(model.state_dict(), self.ckpt_dir / f"{ckpt_prefix}_epoch{epoch}_raw.pt")

                # 保存 EMA 并用 EMA 推理
                if ema is not None:
                    with ema.average_parameters():
                        torch.save(model.state_dict(), self.ckpt_dir / f"{ckpt_prefix}_epoch{epoch}_ema.pt")
                        self.do_inference(model, ds_infer, epoch, stage_name)
                else:
                    self.do_inference(model, ds_infer, epoch, stage_name)

        # 保存 final：维持原 final_name 给下游使用（优先 EMA）
        torch.save(model.state_dict(), self.ckpt_dir / f"saf_final_reflowed_{tag}_raw.pt")
        if ema is not None:
            with ema.average_parameters():
                torch.save(model.state_dict(), self.ckpt_dir / f"saf_final_reflowed_{tag}_ema.pt")
                torch.save(model.state_dict(), self.ckpt_dir / final_name)
        else:
            torch.save(model.state_dict(), self.ckpt_dir / final_name)
        print(f"✅ [Stage 2-{tag}] 训练完成，已保存: {self.ckpt_dir / final_name}")

    def run_all(self):
        if not (self.ckpt_dir / "stage1_final.pt").exists():
            self.run_stage1()

        # 1) 生成（只生成一次）
        gen_marker = self._marker_path("GEN_DONE")
        base_pairs = self._base_pair_files()
        if (not gen_marker.exists()) or (len(base_pairs) == 0):
            self.run_generation()
        self.wait_for_generation_complete()

        # 2) wo_decode 筛选 + Stage2-wo
        self.run_filter_wo_decode()
        wo_dir = self._filter_dir('wo')
        if not (self.ckpt_dir / "saf_final_reflowed_wo.pt").exists():
            self.run_stage2(wo_dir, tag='wo')
        else:
            print("✅ [Stage2-wo] 已检测到最终权重，跳过训练。")

        # 3) w_decode(解码+LPIPS) 筛选 + Stage2-w（不在 wo 的基础上继续训练）
        self.run_filter_w_decode()
        w_dir = self._filter_dir('w')
        if not (self.ckpt_dir / "saf_final_reflowed_w.pt").exists():
            self.run_stage2(w_dir, tag='w')
        else:
            print("✅ [Stage2-w] 已检测到最终权重，跳过训练。")

if __name__ == "__main__":
    trainer = ReflowTrainer()
    try:
        trainer.run_all()
    finally:
        trainer.close()