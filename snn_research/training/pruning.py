# ファイルパス: snn_research/training/pruning.py
# Title: 構造的プルーニング (SBC & Spatio-Temporal) - 実装版
# Description:
# - SBC (Spiking Brain Compression) および 時空間プルーニングの実装。
# - KLダイバージェンスによる時間的冗長性の計算ロジックを実装。
# - time_steps パラメータの動的な更新処理を実装。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, cast, Optional, Type, Iterator
import logging 
from snn_research.core.neurons import AdaptiveLIFNeuron, IzhikevichNeuron
from snn_research.core.snn_core import SNNCore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ... (SBC関連の関数 _get_model_input_keys, _compute_hessian_diag, _compute_saliency, _prune_and_update_weights, apply_sbc_pruning は変更なし、そのまま使用) ...
# 長くなるため、SBC関連部分は省略せず、ユーザー提供コードと同様に完全な形で記述します。

def _get_model_input_keys(model: nn.Module) -> List[str]:
    config_model: Any = None
    if isinstance(model, SNNCore):
        config_model = model.config
    elif hasattr(model, 'config'):
        config_model = model.config # type: ignore[attr-defined]

    if config_model is not None and hasattr(config_model, 'architecture_type'):
        arch_type = config_model.architecture_type
        if arch_type in ["spiking_cnn", "sew_resnet", "hybrid_cnn_snn"]:
            return ["input_images"]
        if arch_type == "tskips_snn":
            return ["input_sequence"]
    return ["input_ids"]

def _compute_hessian_diag(
    model: nn.Module, 
    loss_fn: nn.Module, 
    dataloader: Any,
    max_samples: int = 64
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    logger.info("Computing Hessian matrix diagonal (True Diag Approx.)...")
    params_to_compute: List[nn.Parameter] = []
    param_names: List[str] = []
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad and param.dim() > 1:
            params_to_compute.append(param)
            param_names.append(name)
    if not params_to_compute:
        return {}, {}

    input_keys: List[str] = _get_model_input_keys(model)
    data_iterator: Iterator = iter(dataloader)
    grad_avg: Dict[str, torch.Tensor] = {name: torch.zeros_like(param, device=param.device) for name, param in zip(param_names, params_to_compute)}
    hessian_diag_avg: Dict[str, torch.Tensor] = {name: torch.zeros_like(param, device=param.device) for name, param in zip(param_names, params_to_compute)}
    samples_processed: int = 0
    device: torch.device = next(model.parameters()).device

    while samples_processed < max_samples:
        try:
            batch: Any = next(data_iterator)
            if not isinstance(batch, dict) or "labels" not in batch: continue
            labels: torch.Tensor = batch["labels"].to(device)
            inputs: Dict[str, torch.Tensor] = {k: v.to(device) for k, v in batch.items() if k in input_keys}
            if not inputs: continue
            current_batch_size: int = labels.shape[0]
            for i in range(current_batch_size):
                if samples_processed >= max_samples: break
                sample_inputs: Dict[str, torch.Tensor] = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
                sample_label: torch.Tensor = labels[i].unsqueeze(0)
                model.zero_grad()
                outputs: Tuple[torch.Tensor, ...] = model(**sample_inputs)
                logits: torch.Tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                loss: torch.Tensor = loss_fn(logits.view(-1, logits.size(-1)), sample_label.view(-1)) if logits.dim() == 3 else loss_fn(logits, sample_label)
                first_grads: Tuple[Optional[torch.Tensor], ...] = torch.autograd.grad(loss, params_to_compute, create_graph=True) # type: ignore[assignment]
                for j, (name, param) in enumerate(zip(param_names, params_to_compute)):
                    g_i: Optional[torch.Tensor] = first_grads[j]
                    if g_i is None: continue
                    grad_avg[name] += g_i.detach()
                    H_ii_unsummed: Optional[torch.Tensor] = torch.autograd.grad(g_i, param, grad_outputs=torch.ones_like(g_i), retain_graph=True)[0] # type: ignore[assignment]
                    if H_ii_unsummed is not None: hessian_diag_avg[name] += H_ii_unsummed.detach()
                    else: hessian_diag_avg[name] += (g_i.detach() ** 2)
                samples_processed += 1
        except StopIteration: break
        except Exception as e:
            logger.error(f"Error during Hessian computation: {e}", exc_info=True)
            break

    if samples_processed == 0: return {}, {}
    for name in hessian_diag_avg:
        grad_avg[name] /= samples_processed
        hessian_diag_avg[name] /= samples_processed
        hessian_diag_avg[name] = hessian_diag_avg[name].abs() + 1e-8
    return grad_avg, hessian_diag_avg

def _compute_saliency(param: torch.Tensor, hessian_diag: torch.Tensor) -> torch.Tensor:
    return 0.5 * hessian_diag * (param.data ** 2)

@torch.no_grad()
def _prune_and_update_weights(module: nn.Module, param_name: str, saliency: torch.Tensor, grad: torch.Tensor, hessian_diag: torch.Tensor, amount: float) -> Tuple[int, int]:
    param: torch.Tensor = getattr(module, param_name)
    num_to_prune = int(param.numel() * amount)
    if num_to_prune == 0: return 0, param.numel()
    threshold = torch.kthvalue(saliency.view(-1), k=num_to_prune).values
    mask_keep = saliency > threshold
    mask_prune = ~mask_keep
    correction_term: torch.Tensor = - (grad / hessian_diag)
    delta_w: torch.Tensor = correction_term - param.data
    param.data[mask_prune] += delta_w[mask_prune]
    param.data *= mask_keep.float()
    return int(param.numel() - mask_keep.sum().item()), param.numel()

def apply_sbc_pruning(model: nn.Module, amount: float, dataloader_stub: Any, loss_fn_stub: nn.Module) -> nn.Module:
    if not (0.0 < amount < 1.0): return model
    logger.info(f"--- 🧠 Spiking Brain Compression (SBC/OBC) 開始 (Amount: {amount:.1%}) ---")
    grads_avg, hessian_diagonals = _compute_hessian_diag(model, loss_fn_stub, dataloader_stub)
    if not hessian_diagonals: return model
    total_pruned = 0
    total_params = 0
    model_to_prune: nn.Module = model.model if isinstance(model, SNNCore) and hasattr(model, 'model') else model
    
    for module in model_to_prune.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)) and hasattr(module, 'weight'):
            full_param_name = ""
            for name, mod in model_to_prune.named_modules():
                 if mod is module:
                     full_param_name = f"{name}.weight"
                     break
            if full_param_name in hessian_diagonals:
                pruned, total = _prune_and_update_weights(module, 'weight', _compute_saliency(module.weight, hessian_diagonals[full_param_name]), grads_avg[full_param_name], hessian_diagonals[full_param_name], amount)
                total_pruned += pruned
                total_params += total
    
    if total_params > 0: logger.info(f"--- ✅ SBC (OBC) 完了: {total_pruned}/{total_params} ---")
    return model

# --- ▼▼▼ SNN5改善レポートに基づく時空間プルーニング実装 ▼▼▼ ---

@torch.no_grad()
def _calculate_temporal_redundancy(
    model: nn.Module, 
    dataloader: Any, 
    time_steps: int,
    target_layer_names: Optional[List[str]] = None,
    kl_threshold: float = 0.01
) -> Dict[str, int]:
    """
    KLダイバージェンスに基づき、情報が飽和した冗長なタイムステップを特定する。
    """
    logger.info(f"Calculating temporal redundancy (Actual KL divergence, threshold={kl_threshold})...")
    
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return {}

    model.eval()
    
    # 監視対象レイヤーの特定
    if target_layer_names is None:
        target_layer_names = []
        for name, mod in model.named_modules():
             if isinstance(mod, (AdaptiveLIFNeuron, IzhikevichNeuron)):
                 target_layer_names.append(name)
    
    if not target_layer_names:
        logger.warning("No neuron layers found for temporal analysis.")
        return {}

    # フックによるスパイク活動の収集
    activity_history: Dict[str, List[torch.Tensor]] = {name: [] for name in target_layer_names}
    
    def get_hook(name: str):
        def hook(module, input, output):
            # output[0] がスパイク (B, ...) と仮定
            if isinstance(output, tuple):
                spikes = output[0]
            else:
                spikes = output
            activity_history[name].append(spikes.detach().cpu())
        return hook

    hooks = []
    for name, mod in model.named_modules():
        if name in target_layer_names:
            hooks.append(mod.register_forward_hook(get_hook(name)))

    # データ処理 (最初の数バッチのみで推定)
    num_batches = 5
    iterator = iter(dataloader)
    
    for _ in range(num_batches):
        try:
            batch = next(iterator)
            if isinstance(batch, dict):
                if 'input_ids' in batch: inputs = batch['input_ids'].to(device)
                elif 'input_images' in batch: inputs = batch['input_images'].to(device)
                else: continue
            elif isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                continue
            
            # フック用辞書をクリア
            for name in activity_history: activity_history[name] = []
            
            # 推論実行 (内部でフックが呼ばれ、activity_historyにTステップ分溜まる)
            model(inputs)
            
            break # 1バッチで十分
        except StopIteration:
            break

    for h in hooks: h.remove()

    redundancy_report: Dict[str, int] = {}

    for name, spikes_list in activity_history.items():
        if not spikes_list:
            continue
            
        try:
            spikes_time = torch.stack(spikes_list, dim=0)
            T = spikes_time.shape[0]
            
            # 各ステップの発火率分布 (空間平均)
            # (Time, B, F) -> (Time, F) (バッチ平均)
            if spikes_time.dim() > 2:
                fire_prob = spikes_time.mean(dim=1).float()
            else:
                fire_prob = spikes_time.float() # (Time, F)
            
            # ゼロ除算回避のためのスムージング
            fire_prob = torch.clamp(fire_prob, 1e-6, 1.0 - 1e-6)
            
            saturation_step = T
            
            for t in range(T - 1):
                p = fire_prob[t]
                q = fire_prob[t+1]
                
                # KL(P || Q)
                kl = F.kl_div(q.log(), p, reduction='batchmean')
                
                if kl < kl_threshold:
                    # 変化が閾値以下になったら飽和とみなす
                    saturation_step = t + 1
                    break
            
            redundant_steps = max(0, T - saturation_step)
            redundancy_report[name] = redundant_steps
            
        except Exception as e:
            logger.warning(f"Failed to calc redundancy for {name}: {e}")

    return redundancy_report

@torch.no_grad()
def apply_spatio_temporal_pruning(
    model: nn.Module,
    dataloader: Any,
    time_steps: int,
    spatial_amount: float,
    kl_threshold: float = 0.01
) -> nn.Module:
    """
    時空間プルーニングの実装。
    """
    logger.info(f"--- ⚡️ Spatio-Temporal Pruning 開始 ---")
    
    # 1. 時間プルーニング (Temporal Pruning)
    redundancy_report = _calculate_temporal_redundancy(
        model, dataloader, time_steps, kl_threshold=kl_threshold
    )
    
    avg_redundant_steps: int = 0
    if redundancy_report:
        avg_redundant_steps = int(sum(redundancy_report.values()) / len(redundancy_report))

    new_time_steps = max(1, time_steps - avg_redundant_steps)
    
    if new_time_steps < time_steps:
        logger.info(f"  [Temporal] Reducing time steps from {time_steps} to {new_time_steps}.")
        
        # モデルの属性を更新
        if hasattr(model, 'time_steps'):
            model.time_steps = new_time_steps # type: ignore
        
        # SNNCore内の設定も更新
        if isinstance(model, SNNCore):
             model.config['time_steps'] = new_time_steps
             if hasattr(model.model, 'time_steps'):
                 model.model.time_steps = new_time_steps # type: ignore

    # 2. 空間プルーニング (Spatial Pruning - Magnitude based)
    logger.info("  [Spatial] Applying magnitude-based weight pruning...")
    total_pruned = 0
    total_params = 0
    
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight'):
                param: torch.Tensor = module.weight
                num_to_prune = int(param.numel() * spatial_amount)
                if num_to_prune > 0:
                    threshold = torch.kthvalue(param.data.abs().view(-1), k=num_to_prune).values
                    mask = param.data.abs() > threshold
                    param.data *= mask.float()
                    
                    pruned_count = param.numel() - mask.sum().item()
                    total_pruned += int(pruned_count)
                total_params += param.numel()

    if total_params > 0:
        logger.info(f"  [Spatial] Pruned {total_pruned}/{total_params} weights ({total_pruned/total_params:.2%}).")
    
    logger.info("--- ✅ Spatio-Temporal Pruning 完了 ---")
    return model