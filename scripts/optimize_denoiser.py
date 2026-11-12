# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26", "onnx>=1.18"]
# ///
"""Rewrite the original ARPES denoiser as one portable ONNX model.

Run from the repository root with the original model as INPUT::

    uv run scripts/optimize_denoiser.py INPUT.onnx OUTPUT.onnx

The original model has SHA256
``a765833a830fb328ddcad502a45375b6de9376184657e6bd017c1d5d2ad94e5d``.
The script replaces replicated borders with spatial Slice/Concat operations and
scalar PReLU with equivalent LeakyReLU. It preserves every convolution weight and
bias, all learned activation slopes, and the original model attribution. Image height
and width remain dynamic. No inference or precision conversion is performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ORIGINAL_SHA256 = "a765833a830fb328ddcad502a45375b6de9376184657e6bd017c1d5d2ad94e5d"


def optimize_model(source: onnx.ModelProto) -> onnx.ModelProto:
    """Rewrite the verified original graph without changing its learned function."""
    model = copy.deepcopy(source)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    convolution_parameters = {
        name: initializers[name].SerializeToString()
        for node in model.graph.node
        if node.op_type == "Conv"
        for name in node.input[1:]
    }
    first_start, first_end = "erlab_edge_first_start", "erlab_edge_first_end"
    last_start, last_end = "erlab_edge_last_start", "erlab_edge_last_end"
    spatial_axes = {2: "erlab_edge_height_axis", 3: "erlab_edge_width_axis"}
    model.graph.initializer.extend(
        [
            numpy_helper.from_array(np.array([0], dtype=np.int64), first_start),
            numpy_helper.from_array(np.array([1], dtype=np.int64), first_end),
            numpy_helper.from_array(np.array([-1], dtype=np.int64), last_start),
            numpy_helper.from_array(
                np.array([np.iinfo(np.int64).max], dtype=np.int64), last_end
            ),
            *(
                numpy_helper.from_array(np.array([axis], dtype=np.int64), name)
                for axis, name in spatial_axes.items()
            ),
        ]
    )

    nodes = []
    padding_count = activation_count = 0
    for node in model.graph.node:
        if node.op_type == "Pad":
            mode = next(
                (attr.s for attr in node.attribute if attr.name == "mode"), b"constant"
            )
            if mode != b"edge" or len(node.input) != 2:
                raise ValueError("Expected the original replicated-border Pad")
            # The verified original pads each spatial side by one pixel. These
            # slices retain the axis, including for a one-pixel image.
            current = node.input[0]
            for axis in (2, 3):
                prefix = f"{node.name}_edgecopy_axis{axis}"
                low, high = prefix + "_first", prefix + "_last"
                result = node.output[0] if axis == 3 else prefix + "_padded"
                nodes.extend(
                    [
                        helper.make_node(
                            "Slice",
                            [current, first_start, first_end, spatial_axes[axis]],
                            [low],
                            name=low,
                        ),
                        helper.make_node(
                            "Slice",
                            [current, last_start, last_end, spatial_axes[axis]],
                            [high],
                            name=high,
                        ),
                        helper.make_node(
                            "Concat",
                            [low, current, high],
                            [result],
                            name=prefix + "_concat",
                            axis=axis,
                        ),
                    ]
                )
                current = result
            padding_count += 1
        elif node.op_type == "PRelu":
            slope = numpy_helper.to_array(initializers[node.input[1]])
            if slope.dtype != np.float32 or slope.size != 1:
                raise ValueError("Expected a scalar float32 PReLU slope")
            activation = helper.make_node(
                "LeakyRelu",
                [node.input[0]],
                list(node.output),
                name=node.name,
                alpha=float(slope.item()),
            )
            alpha = next(
                attr.f for attr in activation.attribute if attr.name == "alpha"
            )
            if np.float32(alpha).tobytes() != slope.reshape(()).tobytes():
                raise ValueError("The activation slope changed during conversion")
            nodes.append(activation)
            activation_count += 1
        else:
            nodes.append(copy.deepcopy(node))
    if padding_count != 25 or activation_count != 24:
        raise ValueError("Expected 25 border pads and 24 scalar PReLU activations")

    # Remove the constant subgraphs that constructed the old Pad arguments.
    needed = {value.name for value in model.graph.output}
    kept = []
    for node in reversed(nodes):
        if any(name in needed for name in node.output):
            kept.append(node)
            needed.update(node.input)
    del model.graph.node[:]
    for node in reversed(kept):
        if node.op_type == "Constant":
            # The three division denominators remain unchanged as initializers.
            if len(node.attribute) != 1 or node.attribute[0].name != "value":
                raise ValueError("Expected a tensor-valued Constant")
            tensor = copy.deepcopy(node.attribute[0].t)
            tensor.name = node.output[0]
            model.graph.initializer.append(tensor)
        else:
            model.graph.node.append(node)
    used = {name for node in model.graph.node for name in node.input}
    tensors = [tensor for tensor in model.graph.initializer if tensor.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(tensors)

    for value, spatial_names in (
        (model.graph.input[0], ("height", "width")),
        (model.graph.output[0], ("output_height", "output_width")),
    ):
        for axis, dim in enumerate(value.type.tensor_type.shape.dim):
            dim.ClearField("dim_value")
            dim.ClearField("dim_param")
            if axis < 2:
                dim.dim_value = 1
            else:
                dim.dim_param = spatial_names[axis - 2]
    # ORT infers these dimensions after any CoreML session shape specialization.
    del model.graph.value_info[:]
    actual = {tensor.name: tensor.SerializeToString() for tensor in tensors}
    if any(actual.get(name) != data for name, data in convolution_parameters.items()):
        raise ValueError("A convolution weight or bias changed during conversion")
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Original ARPES denoiser ONNX model")
    parser.add_argument("output", type=Path, help="Destination for the portable model")
    args = parser.parse_args()
    source_bytes = args.input.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != ORIGINAL_SHA256:
        parser.error(f"INPUT must be the original model with SHA256 {ORIGINAL_SHA256}")
    model = optimize_model(onnx.load_model_from_string(source_bytes))
    onnx.save(model, args.output)
    print(
        f"{args.output}: SHA256 {hashlib.sha256(args.output.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
