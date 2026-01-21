import sys
import os
import torch
import gc
import time
import csv
import json
from pathlib import Path
from train import LSFMTrainer

# 🟢 新增：导入评估模块
from eval_lpips import Evaluator as LPIPSEvaluator
from eval_clip import Evaluator as CLIPEvaluator
from eval_vgg import Evaluator as VGGEvaluator

# ==============================================================================
# 🎛️ 实验配置中心
# 这里列出的每个字典代表一次完整的训练任务。
# 字典中的结构与 config.json 完全一致，未列出的参数将使用 config.json 的默认值。
# ==============================================================================

EXPERIMENTS = [
    
    # -------------------------------------------------------------------------
    # Group 1: 稳健基准 (Safe Baseline)
    # 目的：确保代码跑通，建立一个“不好也不坏”的参考系。
    # -------------------------------------------------------------------------
    {
        "name": "E1_Baseline_Safe",
        "description": "【基准】开启OT，SWD权重适中(10)，Patch=7。用于验证系统稳定性。",
        "training": {
            "learning_rate": 1e-4,
            "transfer_loss_weight": 1.0,  # 基础 MSE 权重
            "swd_loss_weight": 10.0,      # 适中的 SWD 约束
            "swd_patch_size": 7,          # 标准 Patch
            "use_ot_reorder": True,       # 开启 OT
            "stage1_epochs": 100,
        }
    },

    # -------------------------------------------------------------------------
    # Group 2: 风格强度的极值探索 (Style Intensity)
    # 目的：验证 SWD 权重是否真的能控制风格化程度？权重过大是否会破坏内容？
    # -------------------------------------------------------------------------
    {
        "name": "E2_HighStyle_SWD50",
        "description": "【强风格】SWD权重提升至50。预期：纹理极强，但内容可能开始扭曲。",
        "training": {
            "swd_loss_weight": 50.0,      # ⬅️ 变量：大幅提升
            "transfer_loss_weight": 1.0,
            "swd_patch_size": 7,
            "use_ot_reorder": True,
            "stage1_epochs": 100,
        }
    },
    {
        "name": "E3_ExtremeStyle_SWD100",
        "description": "【极端风格】SWD权重提升至100。测试模型的崩坏边界。",
        "training": {
            "swd_loss_weight": 100.0,     # ⬅️ 变量：极端提升
            "transfer_loss_weight": 1.0,
            "swd_patch_size": 7,
            "use_ot_reorder": True,
            "stage1_epochs": 100,
        }
    },
    {
        "name": "A3_Ablation_NoSWD",
        "description": "【w/o SWD】关闭 SWD 损失。用于证明 SWD 对分布对齐的关键作用。",
        "training": {
            "transfer_loss_weight": 1.0,
            "swd_loss_weight": 0.0,       # ❌ 关闭 SWD
            "use_ot_reorder": True,
            "stage1_epochs": 100,
        }
    },

    # -------------------------------------------------------------------------
    # Group 3: 纹理粒度测试 (Texture Granularity)
    # 目的：Patch Size 对画质的影响。小 Patch 对应细腻笔触，大 Patch 对应结构风格。
    # -------------------------------------------------------------------------
    {
        "name": "E4_FineTexture_Patch3",
        "description": "【细腻笔触】Patch=3。关注极小范围的纹理统计，预期图像更锐利但可能稍碎。",
        "training": {
            "swd_loss_weight": 20.0,      # 稍微提一点权重配合小 Patch
            "swd_patch_size": 3,          # ⬅️ 变量：极小 Patch
            "swd_patch_stride": 2,        # 采样密一点
            "use_ot_reorder": True,
            "stage1_epochs": 100,
        }
    },

    # -------------------------------------------------------------------------
    # Group 4: OT 有效性验证 (Ablation Study: OT)
    # 目的：这也是论文里必须回答的问题——“OT 真的有用吗？”
    # 对比 E1_Baseline，如果 E1 收敛更快更好，说明 OT 有效。
    # -------------------------------------------------------------------------
    {
        "name": "E5_NoOT_Ablation",
        "description": "【消融实验】关闭 OT。用于对比验证 OT 对收敛速度和几何对齐的贡献。",
        "training": {
            "swd_loss_weight": 10.0,
            "transfer_loss_weight": 1.0,
            "swd_patch_size": 7,
            "use_ot_reorder": False,      # ⬅️ 变量：关闭 OT
            "stage1_epochs": 100,
        }
    },

    # -------------------------------------------------------------------------
    # Group 5: 内容/风格的终极平衡 (The "SOTA" Candidate)
    # 目的：结合高 MSE 权重（保结构）和高 SWD 权重（强风格），试图达到“既要又要”。
    # -------------------------------------------------------------------------
    {
        "name": "E6_Balanced_SOTA",
        "description": "【SOTA候选】高MSE权重(5.0) + 高SWD权重(50.0)。试图在强风格下强行锁住内容结构。",
        "training": {
            "learning_rate": 1e-4,
            "transfer_loss_weight": 5.0,  # ⬅️ 变量：加强内容约束
            "swd_loss_weight": 50.0,      # ⬅️ 变量：加强风格约束
            "swd_patch_size": 5,          # 折中的 Patch
            "use_ot_reorder": True,
            "stage1_epochs": 120,         # 多跑一点
        }
    }
]
# ==============================================================================
# 🟢 修改：评估函数 - 统一输出到实验目录
# ==============================================================================
def _flatten_for_csv(obj, prefix="", sep="."):
    """Flatten nested dicts into a single-level dict suitable for CSV.

    - dict: recurse
    - list/tuple: json-encode
    - other: keep as-is
    """
    flat = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{sep}{k}" if prefix else str(k)
            flat.update(_flatten_for_csv(v, key, sep=sep))
        return flat

    if isinstance(obj, (list, tuple)):
        flat[prefix] = json.dumps(obj, ensure_ascii=False)
        return flat

    flat[prefix] = obj
    return flat


def _summarize_metric_overall(metric_summary, metric_name):
    """Extract overall averages per style as flat keys.

    Expected evaluator format:
      {"overall_average": {"style_0": {"mean": ...}, ...}}
    """
    out = {}
    if not isinstance(metric_summary, dict):
        return out

    if "error" in metric_summary:
        out[f"metrics.{metric_name}.error"] = metric_summary.get("error")
        return out

    overall = metric_summary.get("overall_average")
    if not isinstance(overall, dict):
        # Fallback: flatten everything if structure differs
        return _flatten_for_csv(metric_summary, prefix=f"metrics.{metric_name}")

    for style_key, stats in overall.items():
        if not isinstance(stats, dict):
            continue
        for stat_key in ("mean", "std", "count", "mean_scaled"):
            if stat_key in stats:
                out[f"metrics.{metric_name}.overall_average.{style_key}.{stat_key}"] = stats.get(stat_key)
    return out


def write_experiment_summary_csv(rows, csv_path):
    if not rows:
        print("⚠️  No experiment rows to write.")
        return

    # Stable column order: common keys first, then the rest sorted
    preferred = [
        "experiment_name",
        "description",
        "status",
        "checkpoint_path",
        "experiment_dir",
        "checkpoints_dir",
        "visualizations_dir",
    ]

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())

    remaining = sorted(k for k in all_keys if k not in preferred)
    fieldnames = [k for k in preferred if k in all_keys] + remaining

    csv_path = Path(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n📄 CSV summary saved to: {csv_path.resolve()}")


def run_evaluations(ckpt_path, exp_name, exp_ckpt_dir, config_path="config.json"):
    """
    运行所有三个评估脚本并记录结果
    所有评估结果统一保存到 exp_ckpt_dir/evaluation/ 下
    """
    
    # 🟢 创建统一的评估结果目录
    eval_dir = Path(exp_ckpt_dir) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"📊 Running Evaluations for: {exp_name}")
    print(f"📂 Results will be saved to: {eval_dir}")
    print("="*60)
    
    # 读取配置获取参考目录
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    
    ref_dir = cfg.get("data", {}).get("data_root", None)
    if ref_dir:
        ref_dir = ref_dir.strip('"').strip("'")
    
    target_dir = cfg.get("inference", {}).get("image_path", "").strip('"').strip("'")
    
    # 🟢 汇总结果字典
    results_summary = {
        "experiment_name": exp_name,
        "checkpoint_path": str(ckpt_path),
        "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {}
    }
    
    # 1. LPIPS Evaluation
    try:
        print("\n🔹 [1/3] Running LPIPS Evaluation...")
        if not target_dir:
            raise ValueError("Target directory not configured in config.json")
        lpips_eval = LPIPSEvaluator(str(ckpt_path), config_path)
        lpips_results = lpips_eval.evaluate(target_dir, batch_size=2, save_dir=str(eval_dir))
        results_summary["metrics"]["lpips"] = lpips_results
        del lpips_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ LPIPS Evaluation Complete")
    except Exception as e:
        print(f"❌ LPIPS Evaluation Failed: {e}")
        results_summary["metrics"]["lpips"] = {"error": str(e)}
    
    # 2. CLIP Evaluation
    try:
        print("\n🔹 [2/3] Running CLIP Evaluation...")
        if not target_dir:
            raise ValueError("Target directory not configured in config.json")
        clip_eval = CLIPEvaluator(str(ckpt_path), config_path)
        clip_results = clip_eval.evaluate(target_dir, batch_size=2, save_dir=str(eval_dir))
        results_summary["metrics"]["clip"] = clip_results
        del clip_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ CLIP Evaluation Complete")
    except Exception as e:
        print(f"❌ CLIP Evaluation Failed: {e}")
        results_summary["metrics"]["clip"] = {"error": str(e)}
    
    # 3. VGG Style Evaluation
    try:
        print("\n🔹 [3/3] Running VGG Style Evaluation...")
        vgg_eval = VGGEvaluator(str(ckpt_path), ref_root=ref_dir, config_path=config_path)
        # 🟢 修改：传入保存路径
        vgg_results = vgg_eval.evaluate(bs=1, save_dir=str(eval_dir))
        results_summary["metrics"]["vgg"] = vgg_results
        del vgg_eval
        gc.collect()
        torch.cuda.empty_cache()
        print("✅ VGG Evaluation Complete")
    except Exception as e:
        print(f"❌ VGG Evaluation Failed: {e}")
        results_summary["metrics"]["vgg"] = {"error": str(e)}
    
    # 🟢 保存汇总结果到 JSON
    summary_path = eval_dir / "metrics_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False)
    
    print("\n" + "="*60)
    print(f"📊 All Evaluations Finished for: {exp_name}")
    print(f"📄 Summary saved to: {summary_path}")
    print("="*60)
    
    return results_summary

# ==============================================================================
# 自动化引擎 (Auto-Pilot)
# ==============================================================================
def run_grid_search():
    # 结果总根目录
    ROOT_SAVE_DIR = Path("AutoSearch_Results")
    ROOT_SAVE_DIR.mkdir(exist_ok=True)

    # 汇总表（每个实验一行）
    summary_rows = []
    
    print(f"🚀 Starting Grid Search: {len(EXPERIMENTS)} Experiments Queued.")
    print(f"📂 Root Output: {ROOT_SAVE_DIR.absolute()}")

    for i, exp in enumerate(EXPERIMENTS):
        exp_name = exp['name']
        print("\n" + "#"*60)
        print(f"▶️  [{i+1}/{len(EXPERIMENTS)}] Running Experiment: {exp_name}")
        print(f"ℹ️  Description: {exp.get('description', 'N/A')}")
        print("#"*60)

        # 1. 构造本次实验的专属目录
        exp_dir = ROOT_SAVE_DIR / exp_name
        ckpt_dir = exp_dir / "checkpoints"
        vis_dir = exp_dir / "visualizations"
        
        # 🟢 新增：如果发现旧格式 checkpoint，清理掉防止冲突
        if ckpt_dir.exists():
            old_ckpts = list(ckpt_dir.glob("stage1_epoch*.pt"))
            if old_ckpts:
                # 检查第一个是否为旧格式
                try:
                    test_ckpt = torch.load(old_ckpts[0], map_location='cpu')
                    if 'model_state_dict' not in test_ckpt:
                        print(f"⚠️  Found old format checkpoints in {ckpt_dir.name}")
                        print(f"🗑️  Cleaning up {len(old_ckpts)} old checkpoints...")
                        for old in old_ckpts:
                            old.unlink()
                        print("✅ Cleanup complete")
                except:
                    pass
        
        # 2. 构造配置覆盖 (Override)
        config_override = {
            "checkpoint": {
                "save_dir": str(ckpt_dir)
            },
            "inference": {
                "save_dir": str(vis_dir),
                "num_inference_steps": 4 
            }
        }
        
        # 将用户定义的参数合并进去 (training, model, data 等)
        for k, v in exp.items():
            if k not in ["name", "description"]:
                config_override[k] = v

        start_time = time.time()
        trainer = None
        row = {
            "experiment_name": exp_name,
            "description": exp.get("description", ""),
            "status": "started",
            "experiment_dir": str(exp_dir),
            "checkpoints_dir": str(ckpt_dir),
            "visualizations_dir": str(vis_dir),
        }

        # 记录本次实验的关键 training override 参数，方便对比
        training_overrides = exp.get("training", {}) if isinstance(exp.get("training", {}), dict) else {}
        for k, v in training_overrides.items():
            row[f"training.{k}"] = v

        try:
            # 3. 实例化训练器 (传入覆盖参数)
            trainer = LSFMTrainer(config_override=config_override)
            
            # 4. 运行训练 (只跑 Stage 1 即可快速验证风格)
            trainer.run_stage1()
            
            # 5. 🟢 修复：强制执行一次最终推理
            print("🎨 Running Final Inference...")
            final_ckpt = trainer.ckpt_dir / "stage1_final.pt"
            if final_ckpt.exists():
                final_model = trainer.get_model()
                
                # 🟢 正确加载：先读取checkpoint，提取model_state_dict
                ckpt_data = torch.load(final_ckpt, map_location=trainer.device)
                if 'model_state_dict' in ckpt_data:
                    trainer.safe_load(final_model, ckpt_data['model_state_dict'])
                else:
                    trainer.safe_load(final_model, ckpt_data)
                
                trainer.do_inference(final_model, "final", "stage1_final")
                
                # 清理
                del final_model
                gc.collect()
                torch.cuda.empty_cache()
            
            print(f"✅ Experiment [{exp_name}] Training Completed in {(time.time() - start_time)/60:.1f} mins.")
            
            # 🟢 6. 运行评估脚本 - 传入 ckpt_dir
            if final_ckpt.exists():
                # 先清理训练器释放显存
                del trainer
                gc.collect()
                torch.cuda.empty_cache()
                trainer = None  # 标记已删除

                results_summary = run_evaluations(final_ckpt, exp_name, ckpt_dir)
                row["status"] = "ok"
                row["checkpoint_path"] = str(final_ckpt)

                # 写入 metrics（尽量用 overall 平均）
                metrics = results_summary.get("metrics", {}) if isinstance(results_summary, dict) else {}
                row.update(_summarize_metric_overall(metrics.get("lpips", {}), "lpips"))
                row.update(_summarize_metric_overall(metrics.get("clip", {}), "clip"))
                row.update(_summarize_metric_overall(metrics.get("vgg", {}), "vgg"))
            else:
                row["status"] = "no_final_ckpt"

        except KeyboardInterrupt:
            print("\n🛑 User Interrupted. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Experiment [{exp_name}] Failed!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            row["status"] = "failed"
            row["error"] = str(e)
        finally:
            # 7. 显存清理 (至关重要)
            if trainer:
                del trainer
            gc.collect()
            torch.cuda.empty_cache()
            print("🧹 GPU Memory Cleared.")

            # 记录本实验到汇总表
            summary_rows.append(row)

    print("\n" + "="*60)
    print("🎉 All Experiments Finished!")
    print("="*60)

    # 最终汇总：写 CSV 到“当前目录”(脚本运行目录)
    out_csv = Path.cwd() / "AutoSearch_Results_Summary.csv"
    write_experiment_summary_csv(summary_rows, out_csv)

if __name__ == "__main__":
    run_grid_search()
