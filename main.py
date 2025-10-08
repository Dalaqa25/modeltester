from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import zipfile
import tempfile
import os
import json
from onnx_runner import run_onnx_model

class UploadRequest(BaseModel):
    status: str
    reason: str
    framework_used: str
    task_detection: str

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
async def upload_file(file: UploadFile = File(...), text: str = Form(...)):
    if file.content_type != "application/zip":
        return JSONResponse(status_code=400, content={"message": "File must be a zip file."})

    try:
        data = json.loads(text)
        request = UploadRequest(**data)
    except json.JSONDecodeError:
        # Backward compatibility: assume text is the framework_used string
        request = UploadRequest(status="UNKNOWN", reason="", framework_used=text, task_detection="")

    # Debugging logs
    print(f"Received file: {file.filename}, content_type: {file.content_type}")
    print(f"Received JSON: {request.dict()}")

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

            # Create processed zip
            processed_zip_path = os.path.join(temp_dir, 'processed_model.zip')
            with zipfile.ZipFile(processed_zip_path, 'w') as zip_ref:
                for root, dirs, files in os.walk(temp_dir):
                    for f in files:
                        if f != file.filename and f != 'processed_model.zip':
                            full_path = os.path.join(root, f)
                            arcname = os.path.relpath(full_path, temp_dir)
                            zip_ref.write(full_path, arcname)

            return FileResponse(processed_zip_path, media_type='application/zip', filename='processed_model.zip')

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})