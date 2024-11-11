# Copied from Driss Guessous's PR in PyTorch: https://github.com/pytorch/pytorch/pull/105602

# This file is run to generate the kernel instantiations for the flash_attn kernels
# They are written to several files in order to speed up compilation

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DTYPE_MAP = {
    "fp16": "cutlass::half_t",
    "bf16": "cutlass::bfloat16_t",
}

SM = [80]  # Sm80 kernels support up to
HEAD_DIMENSIONS = [32, 64, 96, 128, 160, 192, 256]
IS_CAUSAL = ["false", "true"]
HAS_RANGE = ["false", "true"]
HAS_TWO_RANGES = ["false", "true"]
KERNEL_IMPL_TEMPLATE_FWD = """#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}>(Flash_fwd_params &params, cudaStream_t stream) {{
    run_mha_fwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}>(params, stream);
}}
"""

KERNEL_IMPL_TEMPLATE_FWD_SPLIT = """#include "flash_fwd_launch_template.h"

template void run_mha_fwd_splitkv_dispatch<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}, {HAS_RANGE}, {HAS_TWO_RANGES}>(Flash_fwd_params &params, cudaStream_t stream);
"""

KERNEL_IMPL_TEMPLATE_BWD = """#include "flash_bwd_launch_template.h"

template<>
void run_mha_bwd_<{DTYPE}, {HEAD_DIM}, {IS_CAUSAL}, {HAS_RANGE}, {HAS_TWO_RANGES}>(Flash_bwd_params &params, cudaStream_t stream) {{
    run_mha_bwd_hdim{HEAD_DIM}<{DTYPE}, {IS_CAUSAL}, {HAS_RANGE}, {HAS_TWO_RANGES}>(params, stream);
}}
"""


@dataclass
class Kernel:
    sm: int
    dtype: str
    head_dim: int
    is_causal: bool
    direction: str
    has_range: bool = False
    has_two_ranges: bool = False

    @property
    def template(self) -> str:
        if self.direction == "fwd":
            return KERNEL_IMPL_TEMPLATE_FWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal,
            )
        elif self.direction == "bwd":
            return KERNEL_IMPL_TEMPLATE_BWD.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal, HAS_RANGE=self.has_range, HAS_TWO_RANGES=self.has_two_ranges
            )
        else:
            return KERNEL_IMPL_TEMPLATE_FWD_SPLIT.format(
                DTYPE=DTYPE_MAP[self.dtype], HEAD_DIM=self.head_dim, IS_CAUSAL=self.is_causal, HAS_RANGE=self.has_range, HAS_TWO_RANGES=self.has_two_ranges
            )

    @property
    def filename(self) -> str:
        return f"flash_{self.direction}_hdim{self.head_dim}_{self.dtype}_{'causal_' if self.is_causal == 'true' else ''}{'range_' if (self.has_range == 'true' and self.has_two_ranges != 'true') else ''}{'2ranges_' if self.has_two_ranges == 'true' else ''}sm{self.sm}.cu"


def get_all_kernels() -> List[Kernel]:
    for direction in ["fwd", "fwd_split", "bwd"]:
        for dtype, head_dim, is_causal, sm in itertools.product(DTYPE_MAP.keys(), HEAD_DIMENSIONS, IS_CAUSAL, SM):
            if direction == "fwd_split":
                for has_range in HAS_RANGE:
                    if has_range == 'true' and is_causal == 'true':
                        continue
                    for has_two_ranges in HAS_TWO_RANGES:
                        if has_two_ranges == 'true' and not has_range == 'true':
                            continue
                        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, is_causal=is_causal, direction=direction, has_range=has_range, has_two_ranges=has_two_ranges)
            elif direction == "bwd":
                for has_range in HAS_RANGE:
                    if has_range == 'true' and is_causal == 'true':
                        continue
                    for has_two_ranges in HAS_TWO_RANGES:
                        if has_two_ranges == 'true' and not has_range == 'true':
                            continue
                        yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, is_causal=is_causal, direction=direction, has_range=has_range, has_two_ranges=has_two_ranges)
            else:
                yield Kernel(sm=sm, dtype=dtype, head_dim=head_dim, is_causal=is_causal, direction=direction)


def write_kernel(kernel: Kernel, autogen_dir: Path) -> None:
    prelude = """// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"\n
"""
    (autogen_dir / kernel.filename).write_text(prelude + kernel.template)


def main(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    for kernel in get_all_kernels():
        write_kernel(kernel, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate_kernels",
        description="Generate the flash_attention kernels template instantiations",
    )
    # Set an optional output directory
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="Where to generate the kernels "
        " will default to the current directory ",
    )
    args = parser.parse_args()
    main(args.output_dir)
