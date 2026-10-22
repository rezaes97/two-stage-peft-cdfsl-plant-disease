# models/proxy_tuning.py
import os
import json
import time
import torch
from collections import deque

from models.model_utils import get_parameters, get_vit_with_adapter_model, feature_extraction_and_fusion
from models.optim_factory import create_optimizer_v2
from models.losses import get_fine_tuning_loss_function, prototype_loss, cross_entropy_loss
from utils import print_trainable_parameters

def _ensure_dir(d):
    if d is None:
        return
    os.makedirs(d, exist_ok=True)

def _write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)

def _save_model_only(save_dir, filename, model_state, meta):
    _ensure_dir(save_dir)
    path = os.path.join(save_dir, filename)
    # Save pure state_dict (no wrapper)
    torch.save(model_state, path)

    # Save metadata JSON alongside the checkpoint so you still have iter/acc/train_args
    meta_out = {
        'iter': int(meta.get('iter', -1)),
        'acc': float(meta.get('acc', 0.0)),
        'train_args': meta.get('train_args', {})
    }
    meta_path = path + ".meta.json"
    try:
        with open(meta_path, 'w') as f:
            json.dump(meta_out, f, indent=2)
    except Exception:
        # best-effort, don't fail on metadata write
        pass

    return path

def _save_best_immediate(save_dir, model_state, meta, i, acc_val, train_args, log_file):
    """
    Save a 'best' checkpoint immediately and update meta.
    We save pure state_dict (via _save_model_only) and update meta['best'].
    """
    if save_dir is None:
        return None
    fname = f"model_best_iter{i:05d}_acc{acc_val:.4f}.pth"
    path = _save_model_only(save_dir, fname, model_state, {'iter': i, 'acc': acc_val, 'train_args': train_args or {}})
    # remove previous best file (if any)
    prev_best = meta.get('best', {}).get('path', None)
    if prev_best and os.path.exists(prev_best) and prev_best != path:
        try:
            os.remove(prev_best)
        except Exception:
            pass
    meta['best'] = {'path': path, 'acc': float(acc_val), 'iter': int(i)}
    # persist meta and log
    try:
        with open(os.path.join(save_dir, 'ckpt_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass
    if log_file:
        try:
            with open(log_file, 'a') as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} NEW BEST iter={i} acc={acc_val:.4f} saved={path}\n")
        except Exception:
            pass
    return path


# --- STREAMING EVALUATION HELPERS ---
import torch.nn.functional as F

def _compute_prototypes_from_loader(model, loader, feature_depth, device):
    """
    Compute class prototypes by streaming through `loader`.
    Returns:
      prototypes_cpu: Tensor [K, D] (normalized, on CPU)
      class_to_idx: dict mapping original label -> prototype row index
      counts: list with counts per prototype index
    """
    model.eval()
    class_to_idx = {}
    sums = []
    counts = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError("loader must yield (images, labels)")
            x = x.to(device)
            feats = feature_extraction_and_fusion(model, x, feature_depth)  # on device
            feats_cpu = feats.detach().cpu()
            y_cpu = y.detach().cpu().view(-1)

            for i, lab in enumerate(y_cpu):
                orig_lbl = int(lab.item())
                if orig_lbl not in class_to_idx:
                    class_to_idx[orig_lbl] = len(class_to_idx)
                    sums.append(torch.zeros(feats_cpu.size(1), dtype=feats_cpu.dtype))
                    counts.append(0)
                idx = class_to_idx[orig_lbl]
                sums[idx] += feats_cpu[i]
                counts[idx] += 1

            # free GPU memory immediately
            del feats, feats_cpu
            torch.cuda.empty_cache()

    if len(sums) == 0:
        return None, {}, []

    sums = torch.stack(sums, dim=0)   # [K, D], on CPU
    counts_t = torch.tensor(counts, dtype=sums.dtype).unsqueeze(1)  # [K,1]
    prototypes = sums / counts_t      # mean per class
    # Normalize prototypes (features already normalized per example; mean may need re-normalization)
    prototypes = F.normalize(prototypes, dim=1)
    return prototypes, class_to_idx, counts

def _eval_with_prototypes(model, loader, prototypes_cpu, class_to_idx, feature_depth, device):
    """
    Evaluate loader by streaming: compute batch features, compare with prototypes,
    return accuracy (float).
    - prototypes_cpu: prototypes on CPU (K, D)
    - class_to_idx: dict orig_label -> row idx in prototypes
    """
    if prototypes_cpu is None or len(class_to_idx) == 0:
        return 0.0

    model.eval()
    # move prototypes to device once
    prototypes = prototypes_cpu.to(device)   # [K, D]
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError("loader must yield (images, labels)")
            x = x.to(device)
            y_cpu = y.detach().cpu().view(-1)

            feats = feature_extraction_and_fusion(model, x, feature_depth)  # on device, [B, D]
            # ensure feats normalized (feature_extraction_and_fusion does normalize already)
            # compute cosine similarity: [B, K]
            sims = F.linear(F.normalize(feats, dim=1), F.normalize(prototypes, dim=1))  # uses linear as dot product
            preds = sims.argmax(dim=1).detach().cpu()  # predicted prototype index

            # map preds back to original labels: build reverse map
            # build once outside loop? small overhead to build here
            # reverse_map: idx -> orig_label
            # create reverse_map now
            # (we compute once; caching slightly more code—kept simple)
            total += y_cpu.size(0)
            # compute correctness: a prediction idx maps to orig label; compare with y_cpu
            # build reverse_map
            if not hasattr(_eval_with_prototypes, "_reverse_map_cache"):
                _eval_with_prototypes._reverse_map_cache = {}
            key = id(prototypes_cpu)
            if key not in _eval_with_prototypes._reverse_map_cache:
                rev = {v: k for k, v in class_to_idx.items()}
                _eval_with_prototypes._reverse_map_cache[key] = rev
            rev_map = _eval_with_prototypes._reverse_map_cache[key]

            for j, pidx in enumerate(preds):
                pidx = int(pidx.item())
                pred_label = rev_map.get(pidx, None)
                if pred_label is None:
                    # unknown prototype index -> treat as incorrect
                    continue
                if pred_label == int(y_cpu[j].item()):
                    correct += 1

            del feats, sims, preds
            torch.cuda.empty_cache()

    return (correct / total) if total > 0 else 0.0

def _eval_proxy_acc_streaming(model, loader, loss_func, feature_depth, device):
    """
    Streaming evaluation for proxy_acc: accumulate logits per batch and compute accuracy.
    Returns accuracy float.
    """
    model.eval()
    loss_func.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x, y = batch[0], batch[1]
            else:
                raise ValueError("loader must yield (images, labels)")
            x = x.to(device)
            y = y.to(device)

            feats = feature_extraction_and_fusion(model, x, feature_depth)
            logits = loss_func.get_logits(feats)   # on device
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            del feats, logits, preds
            torch.cuda.empty_cache()

    return (correct / total) if total > 0 else 0.0

def streaming_evaluate(model, _eval_loader_for_eval, val_loader, feature_depth, loss_func, device, proxy_acc):
    """
    Wrapper that returns (train_acc, val_acc) computed purely streaming.
    If a loader is None, corresponding acc is 0.0.
    """
    train_acc = 0.0
    val_acc = 0.0

    # proxy_acc mode: evaluate classifier logits batch-by-batch on train & val
    if proxy_acc:
        if train_loader is not None:
            train_acc = _eval_proxy_acc_streaming(model, train_loader, loss_func, feature_depth, device)
        if val_loader is not None:
            val_acc = _eval_proxy_acc_streaming(model, val_loader, loss_func, feature_depth, device)
        return train_acc, val_acc

    # prototype mode: compute prototypes from train, then evaluate train and val vs prototypes
    prototypes_cpu, class_to_idx, counts = _compute_prototypes_from_loader(model, train_loader, feature_depth, device) if train_loader is not None else (None, {}, [])
    if prototypes_cpu is None:
        # no prototypes computed -> cannot evaluate prototypes; fallback to zero
        return 0.0, 0.0

    # evaluate train accuracy vs prototypes (classifying each train sample against prototypes)
    if train_loader is not None:
        train_acc = _eval_with_prototypes(model, train_loader, prototypes_cpu, class_to_idx, feature_depth, device)
    if val_loader is not None:
        val_acc = _eval_with_prototypes(model, val_loader, prototypes_cpu, class_to_idx, feature_depth, device)

    return train_acc, val_acc
# --- end streaming helpers ---


def proxy_tuning_streaming(train_loader, val_loader,
                           checkpoint_path,
                           optimizer_type, max_iter,
                           feature_depth,
                           lr, lr_proxy,
                           proxy_acc,
                           embedding_loss_type,
                           save_dir=None, checkpoint_freq=20, keep_last_k=5,
                           train_args=None,
                           device=None,
                           patience_iters=50,
                           eval_every=10,
                           eval_train_loader=None):
    """
    Streaming proxy tuning: trains with minibatches from train_loader.
    - train_loader: yields (images, labels)
    - val_loader: yields (images, labels), can be None
    - checkpoint_path: model init checkpoint path (or None)
    - train_args: dict of hyperparams to save with checkpoints
    - patience_iters: early stopping patience in iterations (stop if no val acc improvement in this many iterations)
    """

    # --- model init
    model, hidden_dim = get_vit_with_adapter_model(ckpt_path=checkpoint_path, **(train_args.get('model_init_kwargs', {}) if train_args else {}))
    print_trainable_parameters(model, loss_func=None)
    device = device or next(model.parameters()).device
    model.to(device)
    _eval_loader_for_eval = eval_train_loader if eval_train_loader is not None else train_loader

    # infer number of train classes automatically
    num_train_classes = None
    ds = getattr(train_loader, 'dataset', None)
    if ds is not None:
        # common dataset attributes
        if hasattr(ds, 'train_class_to_idx'):
            num_train_classes = len(ds.train_class_to_idx)
        elif hasattr(ds, 'class_to_idx'):
            num_train_classes = len(ds.class_to_idx)
        elif hasattr(ds, 'classes'):
            num_train_classes = len(ds.classes)
        elif hasattr(ds, 'train_files'):
            try:
                num_train_classes = len(set(lbl for _, lbl in ds.train_files))
            except Exception:
                num_train_classes = None
    if num_train_classes is None:
        # fallback: try to scan loader once (only labels, not images)
        label_set = set()
        for batch in _eval_loader_for_eval:
            _, y = batch[0], batch[1]
            label_set.update(int(v) for v in y.view(-1).tolist())
        num_train_classes = len(label_set)

    # create a dummy support_y tensor (1..num_classes) to satisfy existing loss init that expects support_y
    support_y_for_init = torch.arange(num_train_classes, dtype=torch.long, device=device)

    loss_func = get_fine_tuning_loss_function(
        support_y_for_init, feature_depth,
        embedding_loss_type=embedding_loss_type,
        hidden_dim=hidden_dim
    )
    if loss_func is None:
        raise ValueError(f"Unsupported streaming loss {embedding_loss_type!r}; use 'proxy_anchor' or 'proxy_anchor_custom'.")
    loss_func = loss_func.to(device)

    # optimizers
    parameters = get_parameters(model)
    optimizer = create_optimizer_v2(parameters, opt=optimizer_type, lr=lr)
    optimizer_anchor = create_optimizer_v2(list(loss_func.parameters()), opt=optimizer_type, lr=lr_proxy)

    model.train(); loss_func.train()
    scaler = torch.amp.GradScaler(device.type if hasattr(device, 'type') else 'cuda', init_scale=2**8)

    # checkpoint bookkeeping
    if save_dir is not None:
        _ensure_dir(save_dir)
        meta_path = os.path.join(save_dir, 'ckpt_meta.json')
        meta = _read_json(meta_path) or {}
        meta.setdefault('last', None)
        meta.setdefault('best', {'path': None, 'acc': -1.0, 'iter': -1})
        meta.setdefault('lastK', [])
        train_args_file = os.path.join(save_dir, 'train_args.json')
        _write_json(train_args_file, train_args or {})
        log_file = os.path.join(save_dir, 'training.log')
        with open(log_file, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} starting proxy_tuning_streaming; max_iter={max_iter}, lr={lr}, lr_proxy={lr_proxy}, optimizer={optimizer_type}, checkpoint_freq={checkpoint_freq}, patience_iters={patience_iters}\n")
    else:
        meta = {'best': {'path': None, 'acc': -1.0, 'iter': -1}}
        log_file = None

    # Training loop variables
    train_iter = iter(train_loader)
    # track last improvement iteration (based on validation acc)
    last_improve_iter = int(meta['best'].get('iter', -1)) if meta and isinstance(meta.get('best', None), dict) else -1

    # If there was an existing best acc loaded from meta, ensure the value is float
    try:
        meta_best_acc = float(meta['best'].get('acc', -1.0))
    except Exception:
        meta_best_acc = -1.0
        meta['best'] = {'path': None, 'acc': meta_best_acc, 'iter': -1}

    # optional container to hold last quick-eval val acc so we don't need to recompute unnecessarily
    last_quick_eval_val_acc = None

    for i in range(max_iter):
        model.train(); loss_func.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x_batch, y_batch = batch[0], batch[1]
        else:
            raise ValueError("train_loader must yield (images, labels)")

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad(); optimizer_anchor.zero_grad()
        with torch.amp.autocast(device.type if hasattr(device, 'type') else 'cuda'):
            feats = feature_extraction_and_fusion(model, x_batch, feature_depth)
            loss = loss_func(feats, y_batch)

        scaler.scale(loss).backward()

        # unscale and clip
        scaler.unscale_(optimizer)
        scaler.unscale_(optimizer_anchor)

        torch.nn.utils.clip_grad_norm_(get_parameters(model),    max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(loss_func.parameters(),  max_norm=1.0)

        # step optimizers (handle AMP internals robustly)
        try:
            if scaler._per_optimizer_states[id(optimizer)].get("found_inf_per_device"):
                scaler.step(optimizer)
            else:
                optimizer.step()
        except Exception:
            optimizer.step()

        try:
            if scaler._per_optimizer_states[id(optimizer_anchor)].get("found_inf_per_device"):
                scaler.step(optimizer_anchor)
            else:
                optimizer_anchor.step()
        except Exception:
            optimizer_anchor.step()

        # update scaler if any inf flags present
        if any(state.get("found_inf_per_device", False)
               for state in scaler._per_optimizer_states.values()):
            scaler.update()

        # quick logging + streaming evaluation every 10 iterations (no full checkpoint save)
        if i % eval_every  == 0:
            print(f"Iter {i}: Loss={loss.item():.4f}", flush=True)

            model.eval(); loss_func.eval()
            with torch.no_grad():
                train_acc, val_acc = streaming_evaluate(model, _eval_loader_for_eval, val_loader, feature_depth, loss_func, device, proxy_acc)

            # store last quick eval value
            last_quick_eval_val_acc = float(val_acc)

            # If this quick-eval produced an improved val acc, update meta.best and save best immediately
            if meta is not None:
                current_best = float(meta['best'].get('acc', -1.0))
                if val_acc > current_best + 1e-12:
                    last_improve_iter = i
                    meta['best'] = {'path': meta['best'].get('path', None), 'acc': float(val_acc), 'iter': i}
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} quick-eval improvement iter={i} val_acc={val_acc:.4f}\n")
                    # Save the new best model immediately (pure state_dict) and update meta['best'].path
                    try:
                        saved = _save_best_immediate(save_dir, model.state_dict(), meta, i, val_acc, train_args or {}, log_file)
                        print(f"[NEW-BEST-SAVED] iter={i} val_acc={val_acc:.4f} -> {saved}", flush=True)
                    except Exception as e:
                        print(f"[WARN] failed to save immediate best at iter={i}: {e}", flush=True)

            print(f"EVAL @iter {i}: TRAIN_ACC={train_acc:.4f} VAL_ACC={val_acc:.4f} (proxy_acc={proxy_acc})", flush=True)

            # Early stopping check
            if val_loader is not None and meta is not None and meta['best'].get('iter', -1) >= 0:
                if (i - last_improve_iter) >= int(patience_iters):
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} early-stopping at iter={i}; no val improvement for {patience_iters} iterations (best_iter={meta['best']['iter']}, best_acc={meta['best']['acc']:.4f})\n")
                    print(f"[EARLY-STOP] No val improvement for {patience_iters} iterations. Stopping at iter={i}. Best iter={meta['best']['iter']}, best_acc={meta['best']['acc']:.4f}", flush=True)
                    if save_dir is not None:
                        with open(os.path.join(save_dir, 'ckpt_meta.json'), 'w') as f:
                            json.dump(meta, f, indent=2)
                    return


        # checkpoint & (more thorough) evaluation only at checkpoint_freq or at last iteration
        if save_dir is not None and (i % checkpoint_freq == 0 or i == max_iter - 1):
            model.eval(); loss_func.eval()
            with torch.no_grad():
                train_acc, val_acc = streaming_evaluate(model, _eval_loader_for_eval, val_loader, feature_depth, loss_func, device, proxy_acc)
                acc_val = float(val_acc)

            # update meta.best if this checkpoint produced a new improvement
            if meta is not None:
                if acc_val > float(meta['best'].get('acc', -1.0)) + 1e-12:
                    meta['best'] = {'path': None, 'acc': float(acc_val), 'iter': i}
                    last_improve_iter = i

            # Decide whether to actually save files: only save if best or final
            is_best_checkpoint = (meta is not None and int(meta['best'].get('iter', -1)) == i)
            is_last_iter = (i == max_iter - 1)
            ckpt_meta = {'iter': i, 'acc': float(acc_val), 'train_args': train_args or {}}
            fname = f"model_iter{i:05d}_acc{acc_val:.4f}.pth"
            saved_path = None

            if is_best_checkpoint or is_last_iter:
                saved_path = _save_model_only(save_dir, fname, model.state_dict(), ckpt_meta)
                if is_best_checkpoint:
                    prev_best = meta['best'].get('path', None)
                    if prev_best and os.path.exists(prev_best) and prev_best != saved_path:
                        try: os.remove(prev_best)
                        except Exception: pass
                    meta['best']['path'] = saved_path
                    meta['best']['acc'] = float(acc_val)
                    meta['best']['iter'] = int(i)
                if is_last_iter:
                    last_path = os.path.join(save_dir, 'last.pth')
                    torch.save(model.state_dict(), last_path)
                    try:
                        with open(last_path + ".meta.json", 'w') as f:
                            json.dump(ckpt_meta, f, indent=2)
                    except Exception:
                        pass
                    meta['last'] = last_path
                if 'lastK' in meta:
                    meta['lastK'].append(saved_path)
                    while keep_last_k is not None and len(meta['lastK']) > keep_last_k:
                        old = meta['lastK'].pop(0)
                        try:
                            if os.path.exists(old): os.remove(old)
                        except Exception: pass
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} checkpoint iter={i} acc={acc_val:.4f} saved={saved_path}\n")
                print(f"[CHECKPOINT] iter={i} acc={acc_val:.4f} saved to {saved_path}; best={meta['best']['acc']:.4f}", flush=True)
            else:
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} evaluated iter={i} acc={acc_val:.4f} (no checkpoint save)\n")
                print(f"[EVAL-ONLY] iter={i} acc={acc_val:.4f} (not saved; best={meta['best']['acc']:.4f})", flush=True)

            # persist meta and log
            if save_dir is not None:
                with open(os.path.join(save_dir, 'ckpt_meta.json'), 'w') as f:
                    json.dump(meta, f, indent=2)

            # Early stopping check here as well
            if val_loader is not None and meta is not None and meta['best'].get('iter', -1) >= 0:
                if (i - int(meta['best']['iter'])) >= int(patience_iters):
                    if log_file:
                        with open(log_file, 'a') as f:
                            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} early-stopping at iter={i}; no val improvement for {patience_iters} iterations (best_iter={meta['best']['iter']}, best_acc={meta['best']['acc']:.4f})\n")
                    print(f"[EARLY-STOP] No val improvement for {patience_iters} iterations. Stopping at iter={i}. Best iter={meta['best']['iter']}, best_acc={meta['best']['acc']:.4f}", flush=True)
                    if save_dir is not None:
                        with open(os.path.join(save_dir, 'ckpt_meta.json'), 'w') as f:
                            json.dump(meta, f, indent=2)
                    return


    # final meta write
    if save_dir is not None:
        with open(os.path.join(save_dir, 'ckpt_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    return



def run_proxy_tuning_episode(
        support_x, query_x, support_y, query_y,
        optimizer_type, max_iter, feature_depth, lr, lr_proxy,
        proxy_acc, embedding_loss_type, checkpoint_path, checkpoint_key="auto",
        tuning_depth=1, ssf_init=False, houl_adapter_reduction=32,
        houl_adapter_dropout=0.0, lora_r=8, lora_alpha=1.0,
        lora_dropout=0.05, lora_target_modules=None,
        base_adapters=None, tuning_adapters=None, device=None):
    """Fast, independent proxy adaptation for one few-shot episode.

    Every call reconstructs the model from the same offline checkpoint, trains only
    the selected online adapters and ProxyAnchor parameters on support images, and
    reports query accuracy. No query image is used for gradient updates.
    """
    init_kwargs = {
        "tuning_depth": int(tuning_depth),
        "ssf_init": bool(ssf_init),
        "houl_adapter_reduction": int(houl_adapter_reduction),
        "houl_adapter_dropout": float(houl_adapter_dropout),
        "lora_r": int(lora_r),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_target_modules": lora_target_modules or ["qkv"],
        "base_adapters": base_adapters or [],
        "tuning_adapters": tuning_adapters or ["ssf"],
    }
    model, hidden_dim = get_vit_with_adapter_model(
        ckpt_path=checkpoint_path, checkpoint_key=checkpoint_key, **init_kwargs
    )
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(device)
    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)

    trainable = get_parameters(model)
    if not trainable:
        raise RuntimeError("No trainable online adapter parameters. Check tuning_adapters and tuning_depth.")

    loss_func = get_fine_tuning_loss_function(
        support_y, feature_depth, embedding_loss_type=embedding_loss_type,
        hidden_dim=hidden_dim
    )
    if loss_func is None:
        raise ValueError(
            f"Unsupported episodic loss {embedding_loss_type!r}; use proxy_anchor or proxy_anchor_custom."
        )
    loss_func = loss_func.to(device)
    anchor_parameters = list(loss_func.parameters())
    if not anchor_parameters:
        raise RuntimeError(f"Loss {embedding_loss_type!r} has no trainable ProxyAnchor parameters.")

    optimizer = create_optimizer_v2(trainable, opt=optimizer_type, lr=float(lr))
    optimizer_anchor = create_optimizer_v2(anchor_parameters, opt=optimizer_type, lr=float(lr_proxy))
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    loss_func.train()
    for _ in range(int(max_iter)):
        optimizer.zero_grad(set_to_none=True)
        optimizer_anchor.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            features = feature_extraction_and_fusion(model, support_x, feature_depth)
            loss = loss_func(features, support_y)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.unscale_(optimizer_anchor)
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(anchor_parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.step(optimizer_anchor)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(anchor_parameters, max_norm=1.0)
            optimizer.step()
            optimizer_anchor.step()

    model.eval()
    loss_func.eval()
    with torch.no_grad():
        support_features = feature_extraction_and_fusion(model, support_x, feature_depth)
        query_features = feature_extraction_and_fusion(model, query_x, feature_depth)
        if proxy_acc:
            _, stats, _ = cross_entropy_loss(loss_func.get_logits(query_features), query_y)
        else:
            _, stats, _ = prototype_loss(
                support_features, support_y, query_features, query_y
            )

    del support_features, query_features, model, loss_func, optimizer, optimizer_anchor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return stats
