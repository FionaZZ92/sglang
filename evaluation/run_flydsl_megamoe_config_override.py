#!/usr/bin/env python3
"""Run FlyDSL's MegaMoE benchmark with temporary Stage1/Stage2 overrides.

This is a tuning helper only.  The production configuration remains selected by
``kernels.mega_moe.mega_moe_config`` until a measured winner is committed there.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from dataclasses import replace
from pathlib import Path

import kernels.mega_moe.mega_moe as mega_moe_impl


def _json_env(name: str) -> dict[str, object]:
    value = os.environ.get(name, "").strip()
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return parsed


stage1_override = _json_env("FLYDSL_MEGA_MOE_STAGE1_OVERRIDE")
stage2_override = _json_env("FLYDSL_MEGA_MOE_STAGE2_OVERRIDE")
combine_override = _json_env("FLYDSL_MEGA_MOE_COMBINE_OVERRIDE")
original_resolve = mega_moe_impl.resolve_mega_moe_config
original_init = mega_moe_impl.MegaMoEV2.__init__


def resolve_with_override(*args, **kwargs):
    config = original_resolve(*args, **kwargs)
    stage1 = replace(config.stage1, **stage1_override)
    stage2 = replace(config.stage2, **stage2_override)
    return replace(config, stage1=stage1, stage2=stage2)


mega_moe_impl.resolve_mega_moe_config = resolve_with_override


def init_with_override(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if combine_override:
        expected = {"block_num", "warp_num_per_block"}
        unknown = combine_override.keys() - expected
        if unknown:
            raise ValueError(f"Unknown combine override fields: {sorted(unknown)}")
        if "block_num" in combine_override:
            self.comb_cfg.combine_block_num = int(combine_override["block_num"])
        if "warp_num_per_block" in combine_override:
            self.comb_cfg.combine_warp_num_per_block = int(
                combine_override["warp_num_per_block"]
            )


mega_moe_impl.MegaMoEV2.__init__ = init_with_override

if int(os.environ.get("RANK", "0")) == 0:
    print(
        "[TUNING-OVERRIDE] "
        f"stage1={json.dumps(stage1_override, sort_keys=True)} "
        f"stage2={json.dumps(stage2_override, sort_keys=True)} "
        f"combine={json.dumps(combine_override, sort_keys=True)}",
        flush=True,
    )

target = os.environ.get(
    "FLYDSL_MEGA_MOE_BENCHMARK",
    str(
        Path(mega_moe_impl.__file__).resolve().parents[2]
        / "tests/kernels/test_mega_moe_v2.py"
    ),
)
sys.argv[0] = target
runpy.run_path(target, run_name="__main__")
