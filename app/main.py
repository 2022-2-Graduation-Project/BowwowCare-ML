# Reference: https://medium.com/@mingc.me/deploying-pytorch-model-to-production-with-fastapi-in-cuda-supported-docker-c161cca68bb8
import os
import sys
import io

import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import torch
import torchvision.models as models
from torchvision import transforms

from config import CONFIG

# Initialize API Server
app = FastAPI(
    title="ML Model",
    description="Description of the ML Model",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)
# python main.py or uvicorn main:app --workers 2 —-port 8080 —-log-config log.ini -—reload

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model
    model = models.resnet18()
    model.load_state_dict(torch.load(
        CONFIG['MODEL_PATH'], map_location=torch.device(CONFIG['DEVICE'])))
    model.eval()

    # add model and other preprocess tools too app state
    app.package = {
        "model": model
    }


@app.post('/api/v1/predict')
async def do_predict(file: UploadFile=File(...)):
    """
    Perform prediction on input data
    """

    logger.info('API predict called')

    # prepare input data
    request_object_content = await file.read()
    image = Image.open(io.BytesIO(request_object_content))

    X = transforms.ToTensor()(image).unsqueeze_(0)
    # X = image

    model = app.package['model']
    with torch.no_grad():
        # move tensor to device
        X = X.to(CONFIG['DEVICE'])

        # run model
        y_pred = model(X)

    # convert result to a numpy array on CPU
    y_pred = y_pred.cpu().numpy()

    # run model inference
    y = y_pred[0]

    # generate prediction based on probablity
    pred = ['angry', 'happy', 'relaxed', 'sad'][y.argmax()]

    # round probablities for json
    y = y.tolist()
    y = list(map(lambda v: round(v, ndigits=CONFIG['ROUND_DIGIT']), y))

    # prepare json for returning
    results = {
        'angry': y[0],
        'happy': y[1],
        'relaxed': y[2],
        'sad': y[3],
        'pred': pred
    }

    logger.info(f'results: {results}')

    return {
        "error": False,
        "results": results
    }


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="0.0.0.0", port=8080,
                reload=True, log_config="log.ini"
                )
