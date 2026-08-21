"""Evaluate official PBT checkpoints on pretraining-unseen CALB and NA-ion.

The released checkpoints are trained on ``MIX_large``, which excludes CALB,
Zn-ion and NA-ion.  Only cells represented in the released prompt-embedding
bundle can be evaluated without regenerating embeddings.  Evaluation uses one
sample per cell at exactly H visible cycles.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from transformers import LlamaConfig


ROOT = Path(__file__).resolve().parents[2]
PBT_ROOT = ROOT / "external" / "PBT"
sys.path.insert(0, str(PBT_ROOT))

from BatteryLifeLLMUtils.configuration_BatteryLifeLLM import BatteryElectrochemicalConfig, BatteryLifeConfig
from data_provider.data_loader import Dataset_PBT, my_collate_fn
from data_provider.data_split_recorder import split_recorder
from data_provider.gate_masker import gate_masker
from models.PBT import Model


OUTPUT = Path(__file__).with_name("pbt_official_unseen_eval.json")
SEEDS = (2021, 42, 2024)


def checkpoint(seed):
    return next((PBT_ROOT / "pretrained" / "extracted" / "PBTs").glob(f"*seed{seed}-100"))


def available_prompt_names():
    names = set()
    for split in ("training", "validation", "testing"):
        path = PBT_ROOT / "pretrained" / "extracted" / f"{split}_DKP_embed_all_Llama.pkl"
        names.update(pickle.load(open(path, "rb")))
    return names


def eligible_files(dataset):
    folder = "NA-ion" if dataset == "NAion" else dataset
    label_name = "NA-ion" if dataset == "NAion" else dataset
    labels = json.load(open(ROOT / "data" / "batterylife_processed" / "Life labels" / f"{label_name}_labels.json"))
    prompts = available_prompt_names()
    files = []
    for path in sorted((ROOT / "data" / "batterylife_processed" / folder).glob("*.pkl")):
        if path.name in labels and float(labels[path.name]) > 100 and path.stem in prompts:
            files.append(path.name)
    return files


def make_dataset(args, dataset, files, scaler, horizon):
    if dataset == "CALB":
        split_recorder.CALB_val_files = list(files)
    elif dataset == "NAion":
        split_recorder.NAion_2021_val_files = list(files)
    elif dataset == "HNEI":
        split_recorder.HNEI_val_files = list(files)
    else:
        raise KeyError(dataset)
    args.dataset = dataset
    return Dataset_PBT(
        args,
        flag="val",
        label_scaler=scaler,
        eval_cycle_min=horizon,
        eval_cycle_max=horizon,
        temperature2mask=gate_masker.MIX_large_temperature2mask,
        format2mask=gate_masker.MIX_large_format2mask,
        cathodes2mask=gate_masker.MIX_large_cathodes2mask,
        anode2mask=gate_masker.MIX_large_anode2mask,
        ion2mask=None,
    )


def evaluate(seed, dataset, horizon):
    ckpt = checkpoint(seed)
    args_dict = json.load(open(ckpt / "args.json"))
    args_dict["root_path"] = str(PBT_ROOT / "dataset_v12")
    args_dict["num_workers"] = 0
    args_dict["batch_size"] = 8
    args = BatteryElectrochemicalConfig(args_dict).get_configs()
    scaler = joblib.load(ckpt / "label_scaler")
    files = eligible_files(dataset)
    data = make_dataset(args, dataset, files, scaler, horizon)
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0, collate_fn=my_collate_fn)

    config = BatteryLifeConfig(BatteryElectrochemicalConfig(args_dict), LlamaConfig(hidden_size=args_dict["d_llm"]))
    model = Model(config)
    load_model(model, ckpt / "model.safetensors", strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    mean, std = float(scaler.mean_[0]), float(scaler.scale_[0])
    rows = []
    with torch.inference_mode():
        for batch in loader:
            (
                curves,
                masks,
                labels,
                _,
                names,
                embeddings,
                _,
                cathode,
                temperature,
                form,
                anode,
                ion,
                combined,
                _,
            ) = batch
            output = model(
                curves.to(device),
                masks.to(device),
                DKP_embeddings=embeddings.to(device),
                cathode_masks=cathode.to(device),
                temperature_masks=temperature.to(device),
                format_masks=form.to(device),
                anode_masks=anode.to(device),
                ion_type_masks=ion.to(device),
                combined_masks=combined.to(device),
            )[0]
            predictions = output.detach().float().cpu().numpy().reshape(-1) * std + mean
            truths = labels.numpy().reshape(-1) * std + mean
            for name, prediction, truth in zip(names, predictions, truths):
                error = abs(float(prediction) - float(truth))
                rows.append(
                    {
                        "file": name,
                        "prediction": float(prediction),
                        "truth": float(truth),
                        "abs_error": error,
                        "ape": 100.0 * error / max(float(truth), 1.0),
                    }
                )
    if len(rows) != len(files) or len({r["file"] for r in rows}) != len(files):
        raise RuntimeError(f"Expected one prediction per cell, got {len(rows)} for {len(files)} files")
    return rows


def summarize(rows):
    return {
        "n_cells": len(rows),
        "mae": float(np.mean([r["abs_error"] for r in rows])),
        "mape": float(np.mean([r["ape"] for r in rows])),
        "median_ae": float(np.median([r["abs_error"] for r in rows])),
    }


def main():
    original_cwd = Path.cwd()
    os.chdir(PBT_ROOT)
    try:
        if OUTPUT.exists():
            output = json.load(open(OUTPUT))
        else:
            output = {
                "source": "official Ruifeng-Tan/PBT checkpoints, Zenodo record 17972780",
                "checkpoint_md5": "7a2f59d027cf13c3e608d3b20a9b5291",
                "pretraining_dataset": "MIX_large (excludes CALB and NA-ion; includes HNEI)",
                "protocol": "one prediction per cell at exactly H cycles; no target labels or target-domain model fitting",
                "runs": [],
            }
        completed = {(r["dataset"], r["horizon"], r["checkpoint_seed"]) for r in output["runs"]}
        for dataset in ("CALB", "NAion", "HNEI"):
            for horizon in (10, 20, 50):
                for seed in SEEDS:
                    if (dataset, horizon, seed) in completed:
                        continue
                    rows = evaluate(seed, dataset, horizon)
                    output["runs"].append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "checkpoint_seed": seed,
                            "summary": summarize(rows),
                            "rows": rows,
                        }
                    )
        output["aggregate"] = {}
        for dataset in ("CALB", "NAion", "HNEI"):
            for horizon in (10, 20, 50):
                runs = [r for r in output["runs"] if r["dataset"] == dataset and r["horizon"] == horizon]
                output["aggregate"][f"{dataset}_h{horizon}"] = {
                    "n_cells": runs[0]["summary"]["n_cells"],
                    "mae_mean": float(np.mean([r["summary"]["mae"] for r in runs])),
                    "mae_std": float(np.std([r["summary"]["mae"] for r in runs])),
                    "mape_mean": float(np.mean([r["summary"]["mape"] for r in runs])),
                    "mape_std": float(np.std([r["summary"]["mape"] for r in runs])),
                }
        OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(output["aggregate"], indent=2))
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
