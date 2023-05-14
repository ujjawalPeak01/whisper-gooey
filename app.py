import json
import numpy as np
import torch
from transformers import pipeline


class InferlessPythonModel:
    def initialize(self):
        self.generator = pipeline("automatic-speech-recognition", model="vasista22/whisper-hindi-large-v2",device=0)

    def infer(self, inputs):
        audio_url =  inputs["audio_url"]
        pipeline_output = self.generator(audio_url)
        generated_txt = pipeline_output["text"]
        data = { "generated_txt" : generated_txt } 
        return data

    def finalize(self):
        self.pipe = None
