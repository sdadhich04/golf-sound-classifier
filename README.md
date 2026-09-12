# Golf Sound Classifier

TinyML audio-classification project for EE 446 at the University of Washington. The project classifies golf impact sounds into three labels:

- `Good`
- `Ground`
- `Top`

The included report and slides describe deployment on an Arduino Nano 33 BLE Sense using Edge Impulse. The deployment export in this checkout is an Edge Impulse C++ library for project `Golf` (`project_id` 1027938) owned by `sdadhich`.

## Approach

The offline notebook, `Golf_TinyML_Pipeline.ipynb`, implements a companion training and evaluation pipeline:

1. Load labeled WAV clips from a local `Golf dataset/` folder.
2. Infer labels from filename prefixes: `good`, `grnd`, and all other names as `Top`.
3. Downmix audio to mono and resample to 16 kHz.
4. Split clips into train and test sets before augmentation.
5. Apply training-only augmentation with small time shifts, light noise, and gain changes.
6. Extract 40-band mel filterbank energy features from 500 ms windows.
7. Train a small Keras 1D CNN and convert it to an int8 TensorFlow Lite model.

The Edge Impulse deployment archive contains an MFE DSP block, an int8 TensorFlow Lite Micro classifier, and three output categories: `Good`, `Ground`, and `Top`. Its generated metadata lists microphone input at 16 kHz, 8,000 raw samples per model window, 1,960 neural-network input features, and int8 input/output tensors.

## Data

The checkout includes three MP3 recording-session files under `dataset/`:

- `20260528_125335.mp3`
- `20260528_130155.mp3`
- `20260528_131029.mp3`

The labeled WAV clips used by the notebook are not included in this checkout. The saved notebook output shows 76 labeled WAV recordings when run with the expected local data folder: 28 `Good`, 24 `Ground`, and 24 `Top`. To rerun the notebook, provide the labeled WAV clips in a repo-root folder named `Golf dataset/`.

## Repository Contents

- `Golf_TinyML_Pipeline.ipynb` - offline Python/TensorFlow companion pipeline.
- `deployment/golf-cpp-mcu-v2-impulse-3.zip` - Edge Impulse C++ library export with the generated model and SDK files.
- `dataset/*.mp3` - raw recording-session audio files.
- `demo_live_inference.mp4` - demo video artifact.
- `GolfSoundClassifier_EE446_FinalProjectReport.pdf` - final report.
- `Golf_TinyML_EE446.pptx` - final presentation.
- `project_proposal.pdf` - project proposal.

## Hardware and Tools

- Arduino Nano 33 BLE Sense / Nano 33 BLE Sense Rev2, as described in the report and slides.
- Edge Impulse Studio for the deployed impulse, MFE block, quantization, and C++ export.
- TensorFlow/Keras, TensorFlow Lite, librosa, NumPy, soundfile, matplotlib, and scikit-learn in the offline notebook.

## How to Run the Notebook

Install the notebook dependencies:

```bash
pip install numpy soundfile librosa matplotlib tensorflow scikit-learn
```

Place labeled WAV clips in:

```text
Golf dataset/
```

Then open and run:

```text
Golf_TinyML_Pipeline.ipynb
```

The notebook writes an int8 TensorFlow Lite model named `golf_model_int8.tflite`.

## Credits

The project proposal lists the team as Mihir Sharma, Kenzie Kosatria, and Sparsh Dadhich. The final report credits Kenzie with the demo video, offline reproduction notebook, presentation slides, Edge Impulse support, augmentation/evaluation work, and lab report; Mihir with dataset recording, on-device deployment, live swings in the demo, and report review; and Sparsh with leading most Edge Impulse work, model training and quantization, in-class presentation, and report review.

## License

MIT license. See `LICENSE`.
