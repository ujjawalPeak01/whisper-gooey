import json
import numpy as np
import torch
from transformers import pipeline
import cv2


class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline("automatic-speech-recognition", model="vasista22/whisper-hindi-large-v2",device=0)
        url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        response = requests.get(url, stream=True).raw
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        image = cv2.imdecode(arr, -1)
        sift = cv2.SIFT_create()

    def infer(self, inputs):
        audio_url =  inputs["audio_url"]
        task =  inputs["task"]
        language =  inputs["language"]
        return_timestamps =  inputs["return_timestamps"]
        pipeline_output = self.generator(audio_url)
        generated_txt = pipeline_output["text"]
        data = { "transcribed_text" : generated_txt } 
        return data

    def finalize(self):
        self.pipe = None
