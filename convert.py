import torch
import coremltools as ct
import argparse
import sys
import os

try:
    from spandrel import ModelLoader
    from spandrel.architectures.DAT.__arch.DAT import Adaptive_Spatial_Attention
    _HAS_DAT = True
except (ImportError, Exception):
    _HAS_DAT = False

# Patch DAT architecture if available — its sliding-window masks use dynamic shapes
# that break CoreML's MIL compiler if left untreated.
if _HAS_DAT:
    _orig_mask = Adaptive_Spatial_Attention.calculate_mask
    def _static_mask(self, H, W, dtype=None):
        static_H = int(H) if isinstance(H, (int, float)) else 512
        static_W = int(W) if isinstance(W, (int, float)) else 512
        return _orig_mask(self, static_H, static_W, dtype=torch.float32)
    Adaptive_Spatial_Attention.calculate_mask = _static_mask


class OutputScaledModel(torch.nn.Module):
    """
    Wraps a raw PyTorch nn.Module (spandrel_model.model).

    CoreML ImageType input/output contract:
      - Input : arrives as [0, 255] uint8 image from iOS → CoreML scales to [0, 1] via scale=1/255
      - Output: we scale back to [0, 255] so CoreML ImageType renders it correctly

    We trace spandrel_model.model directly (NOT ImageModelDescriptor.__call__)
    because Spandrel's __call__ uses @torch.inference_mode() which breaks JIT.
    The raw model may output values outside [0, 1]; we clamp them (matching Spandrel).
    """
    def __init__(self, raw_model: torch.nn.Module):
        super().__init__()
        self.raw_model = raw_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.raw_model(x)
        out = torch.clamp(out, 0.0, 1.0)
        out = out * 255.0
        return out


def convert(model_path: str, output_path: str, tile_size: int, use_mlprogram: bool) -> bool:
    print(f"\n{'='*60}")
    print(f"Model    : {model_path}")
    print(f"Output   : {output_path}")
    print(f"Tile     : {tile_size}x{tile_size}")
    print(f"Format   : {'mlprogram (FLOAT16)' if use_mlprogram else 'neuralnetwork (FLOAT32)'}")
    print(f"{'='*60}\n")

    # 1. Load with Spandrel
    try:
        from spandrel import ModelLoader
        spandrel_model = ModelLoader().load_from_file(model_path)
    except Exception as e:
        print(f"[Error] Failed to load model: {e}", file=sys.stderr)
        return False

    spandrel_model.eval()
    arch = spandrel_model.architecture.name
    scale = spandrel_model.scale
    print(f"Architecture : {arch}")
    print(f"Scale        : {scale}x\n")

    # 2. Trace with a SMALL dummy input (64x64) for speed.
    # ESRGAN/RRDB models are fully convolutional — the JIT graph is identical
    # regardless of spatial dimensions. Tracing at 512x512 on CPU takes 30-60 min;
    # tracing at 64x64 takes ~10 seconds. CoreML input shape is declared separately.
    trace_size = 64
    trace_input = torch.rand(1, 3, trace_size, trace_size)
    print(f"Tracing with {trace_size}x{trace_size} dummy input (fast trace)...")
    try:
        wrapped = OutputScaledModel(spandrel_model.model)
        wrapped.eval()
        with torch.no_grad():
            traced = torch.jit.trace(wrapped, trace_input)
    except Exception as e:
        print(f"[Error] JIT tracing failed: {e}", file=sys.stderr)
        return False

    # 3. Verify output scale factor using the trace input
    with torch.no_grad():
        trace_out = traced(trace_input)
    detected_scale = trace_out.shape[-1] // trace_size
    print(f"Detected scale : {detected_scale}x")
    print(f"CoreML input   : {tile_size}x{tile_size}")
    print(f"CoreML output  : {tile_size * detected_scale}x{tile_size * detected_scale}\n")

    # 4. Convert to CoreML
    # Declare the REAL tile size here (not the trace size)
    input_type = ct.ImageType(
        name="input",
        shape=(1, 3, tile_size, tile_size),
        scale=1 / 255.0,
        color_layout=ct.colorlayout.RGB,
    )
    output_type = ct.ImageType(
        name="output",
        color_layout=ct.colorlayout.RGB,
    )

    convert_kwargs = dict(
        inputs=[input_type],
        outputs=[output_type],
    )
    if use_mlprogram:
        convert_kwargs["convert_to"] = "mlprogram"
        convert_kwargs["compute_precision"] = ct.precision.FLOAT16
        convert_kwargs["minimum_deployment_target"] = ct.target.iOS16
    else:
        convert_kwargs["convert_to"] = "neuralnetwork"

    try:
        coreml_model = ct.convert(traced, **convert_kwargs)
    except Exception as e:
        print(f"[Error] CoreML conversion failed: {e}", file=sys.stderr)
        return False

    # 5. Metadata
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    coreml_model.author = "coreMLConversions"
    coreml_model.short_description = f"{arch} {scale}x upscaler. Converted from {os.path.basename(model_path)} via spandrel+coremltools."
    coreml_model.version = "1.0"
    coreml_model.user_defined_metadata["scale_factor"] = str(scale)
    coreml_model.user_defined_metadata["tile_size"] = str(tile_size)
    coreml_model.user_defined_metadata["architecture"] = arch

    # 6. Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        coreml_model.save(output_path)
    except Exception as e:
        print(f"[Error] Failed to save model: {e}", file=sys.stderr)
        return False

    print(f"\n[OK] Saved: {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ESRGAN/RRDB/SwinIR upscaling models to CoreML."
    )
    parser.add_argument("--model", required=True, help="Path to .pth or .safetensors file")
    parser.add_argument("--output", required=True, help="Output .mlmodel or .mlpackage path")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Model input tile size in pixels (default: 512)")
    parser.add_argument("--mlprogram", action="store_true",
                        help="Use mlprogram+FLOAT16 format (requires macOS; best for Neural Engine)")
    args = parser.parse_args()

    ok = convert(args.model, args.output, args.tile_size, args.mlprogram)
    sys.exit(0 if ok else 1)
