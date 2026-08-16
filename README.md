# Namibian Sign Language (NSL) Webcam Translator — Starter Project

There is no large public NSL dataset (unlike ASL), so this project is built
around a **train-it-yourself** pipeline: you record your own webcam samples
for the signs you care about, train a lightweight classifier, then run it
live. Start small (a handful of signs / the alphabet / common greetings)
and expand once the pipeline works.

## How it works

1. **MediaPipe's HandLandmarker** extracts 21 3D landmark points per hand
   from each webcam frame — this is the "skeleton" of the hand shape,
   robust to lighting and skin tone. (This uses MediaPipe's current Tasks
   API — the older `mp.solutions.hands` API was removed in recent
   mediapipe releases.) The first time you run `collect_data.py` or
   `translate_live.py`, it will automatically download a small model
   file (`hand_landmarker.task`, ~7 MB) — this needs internet access
   just that once.
2. Landmarks are saved to a CSV, labelled with the sign name.
3. A **RandomForest** classifier learns to map landmark patterns to sign
   labels (fast to train, works well on small datasets, runs on modest
   laptops — relevant if you're developing on lower-spec hardware).
4. The live script re-runs this pipeline per frame and displays the
   predicted sign, smoothing predictions over several frames before adding
   a word to the sentence (reduces flicker/false triggers).

## Setup (VS Code or Spyder)

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

In Spyder: open the project folder via File > Open Project, then run each
script from the toolbar. In VS Code: open the folder, select the venv
interpreter (bottom right), run scripts with the green Run arrow or
`python filename.py` in the integrated terminal.

## Step-by-step usage

### 1. Collect data for each sign

```bash
python collect_data.py
```

Enter a label (e.g. `HELLO`), then press `s` repeatedly while holding the
sign at slightly different angles/positions to build up ~150–300 samples.
Repeat this script for every sign you want recognised (different label
each time — all samples accumulate in `data/landmarks.csv`).

### 2. Train the model

```bash
python train_model.py
```

Prints accuracy/precision per sign and saves the model to `model/`.
If accuracy is low for a sign, go back and collect more/cleaner samples
for it (vary hand angle, distance from camera, lighting).

### 3. Run live translation

```bash
python translate_live.py
```

Shows the predicted sign and confidence, and builds a sentence as you
sign in sequence. Press `c` to clear, `q` to quit.

## Namibian-context notes

- **No assumptions about which sign language variant**: Namibia doesn't
  have one single standardised, widely-documented sign language dataset.
  If possible, work with Deaf Namibian signers or an organisation like the
  Namibia National Association of the Deaf (NNAD) to confirm the signs
  you're recording are actually correct NSL, not borrowed ASL/SASL signs —
  this matters a lot for real usefulness, not just accuracy.
- **Low-bandwidth / offline-first**: everything above runs fully locally,
  no cloud API calls or internet needed after installation — useful given
  data costs and connectivity outside major towns.
- **Modest hardware**: RandomForest + MediaPipe landmarks (rather than
  training a deep CNN on raw video) keeps this runnable on a typical
  student laptop without a GPU.
- **Extend later**: once single-sign recognition works, consider adding
  temporal models (e.g. an LSTM over landmark sequences) for signs that
  involve motion, not just a static hand shape.

## Known limitations

- Single-hand-shape signs only for now — no motion/sequence modelling.
- Accuracy depends heavily on how much and how varied your training data
  is per sign.
- Backgrounds/lighting differences between collection and live use can
  hurt accuracy — collect data in conditions similar to where you'll demo it.
