import os
import onnxruntime
import io
import sys
import onnx
import numpy as np

def run_onnx_model(temp_dir):
    onnx_model_path = None
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            if f.endswith(".onnx"):
                onnx_model_path = os.path.join(root, f)
                break
        if onnx_model_path:
            break

    if not onnx_model_path:
        return None, None, None, "No ONNX model found in the zip file.", None

    print(f"Found ONNX model at: {onnx_model_path}")

    opset_warning = None
    try:
        model = onnx.load(onnx_model_path)
        if model.opset_import:
            opset_version = model.opset_import[0].version
            if opset_version < 7:
                opset_warning = f"Your model uses ONNX opset {opset_version}, which is outdated. Please upgrade to opset 7 or above."
    except Exception as e:
        print(f"Could not check opset version: {e}")


    old_stderr = sys.stderr
    sys.stderr = captured_stderr = io.StringIO()
    
    session = onnxruntime.InferenceSession(onnx_model_path)

    sys.stderr = old_stderr
    warnings_str = captured_stderr.getvalue()
    warnings = warnings_str.strip().split('\n') if warnings_str else []

    print(f"loading model from: {onnx_model_path}")
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]

    # Test inference with dummy data
    try:
        inputs_dict = {}
        for input_info in session.get_inputs():
            shape = input_info.shape
            if shape and len(shape) > 0 and shape[0] == -1:
                # Replace dynamic batch with 1
                shape = (1,) + tuple(shape[1:])
            elif not shape or any(s == -1 for s in shape):
                # Skip if shape is unknown or has other dynamic dims
                print(f"Skipping inference test due to dynamic shape in input {input_info.name}")
                break
            else:
                shape = tuple(shape)
            # Map ONNX types to numpy dtypes
            type_map = {
                'tensor(float)': np.float32,
                'tensor(double)': np.float64,
                'tensor(int32)': np.int32,
                'tensor(int64)': np.int64,
                # Add more if needed
            }
            dtype = type_map.get(str(input_info.type), np.float32)
            dummy_input = np.zeros(shape, dtype=dtype)
            inputs_dict[input_info.name] = dummy_input
        else:
            # Only run if all inputs were created
            session.run(output_names, inputs_dict)
            print("Inference test successful: Model runs with dummy data")
    except Exception as e:
        print(f"Inference test failed: {e}")

    return input_names, output_names, warnings, None, opset_warning