# coreMLConversions

Automated pipeline to convert PyTorch super-resolution model files (`.pth`, `.safetensors`) into optimized CoreML `.mlpackage` files using Apple's `coremltools` running on macOS GitHub Actions.

## Usage

### 1. Drop model files into `models/`

```
models/
  4xLSDIR.pth
  RealESRGAN_x4plus.pth
  myCustomModel.safetensors
```

Push to the repo. GitHub Actions will automatically pick up the new files.

### 2. Wait for the workflow to finish

Go to **Actions → Convert Models to CoreML** in the GitHub UI. The job runs on a `macos-14` Apple Silicon runner.

Conversion per model takes roughly **3–8 minutes** depending on size.

### 3. Download the result

- **As an artifact**: Download the zip from the Actions run (90-day retention).
- **From the repo**: Converted models are committed to `output/` automatically.

### Manual Trigger

Go to **Actions → Convert Models to CoreML → Run workflow** and enter the model filename.

---

## Why macOS?

`coremltools` requires macOS to save `mlprogram` format (`.mlpackage`) — the format that enables FLOAT16 precision and Neural Engine (ANE) acceleration on iPhone/iPad. Windows and Linux runners can only produce the legacy `neuralnetwork` format.

## Output Format

| Feature | This Pipeline |
|---|---|
| Format | `mlprogram` (`.mlpackage`) |
| Precision | FLOAT16 |
| Target | iOS 16+ / macOS 13+ |
| Input | 512×512 RGB image |
| Output | 2048×2048 RGB image (4x scale) |

## Supported Architectures

Any architecture supported by [Spandrel](https://github.com/chaiNNer-org/spandrel), including:

- ✅ ESRGAN / RealESRGAN
- ✅ BSRGAN / MM-RealSR
- ✅ SwinIR / Swin2SR
- ✅ SPAN, ATD, HAT, SCUNet...
- ❌ DAT2 (dynamic shape ops, not yet supported)

## Local Conversion (Windows/Linux — Legacy Format)

```bash
pip install torch coremltools spandrel safetensors
python convert.py --model models/4xLSDIR.pth --output output/4xLSDIR.mlmodel --tile-size 512
# Note: Without --mlprogram, produces legacy neuralnetwork format (.mlmodel)
```
