from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


path = "C:\\Users\\user\\anaconda3\\envs\\nlp\\Lib\\site-packages\\TTS\\.models.json"

model_manager = ModelManager(path)

model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/tacotron2-DDC_ph")
voc_path, voc_config_path, _ = model_manager.download_model(model_item["default_vocoder"])

syn = Synthesizer(tts_checkpoint=model_path, tts_config_path=config_path, vocoder_checkpoint=voc_path, vocoder_config=voc_config_path)

f = open("script.txt", "r")
text = f.read()
f.close()

outputs = syn.tts(text)
syn.save_wav(outputs, "audio.wav")