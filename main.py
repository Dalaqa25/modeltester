from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import zipfile
import tempfile
import os
from onnx_runner import run_onnx_model

class UploadRequest(BaseModel):
    text: str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...), request: UploadRequest = None, text: str = Form(None)):
    if file.content_type != "application/zip":
        return JSONResponse(status_code=400, content={"message": "File must be a zip file."})

    text_value = request.text if request else text
    if not text_value:
        return JSONResponse(status_code=400, content={"message": "Text data is required."})

    # Debugging logs
    source = "JSON" if request else "Form"
    print(f"Received data via: {source}")
    print(f"Received file: {file.filename}, content_type: {file.content_type}")
    print(f"Received text: {text_value}")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, file.filename)
            with open(zip_path, "wb") as buffer:
                buffer.write(await file.read())
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            extracted_files = os.listdir(temp_dir)
            print(f"Temporary directory: {temp_dir}")
            print(f"Extracted files: {extracted_files}")
            
            input_names, output_names, warnings, error, opset_warning = run_onnx_model(temp_dir)

            if opset_warning:
                return JSONResponse(status_code=422, content={"status": "INVALID", "reason": opset_warning})

            if error:
                return JSONResponse(status_code=404, content={"message": error})

            if warnings:
                return JSONResponse(status_code=422, content={"warnings": warnings})

            response_content = {
                "message": "ONNX model loaded successfully",
                "filename": file.filename,
                "text": text_value,
                "extracted_files": extracted_files,
                "model_inputs": input_names,
                "model_outputs": output_names
            }

            return JSONResponse(status_code=200, content=response_content)

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})