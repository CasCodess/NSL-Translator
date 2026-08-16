# Namibian Sign Language (NSL) Webcam Translator

This is my personal project where I am trying to build a webcam based
translator for Namibian Sign Language (NSL). The idea is simple: you show
a sign to your webcam, and the program tries to recognize which sign it
is and shows you the word on screen.

I started this project because there is no big public dataset for NSL
like there is for American Sign Language (ASL). So instead of downloading
a ready made dataset, I built a pipeline where I record my own hand
signs using my webcam, train a small model on them, and then use that
model to recognize signs live.

## Project breakdown (SMART analysis)

**Specific**
The goal is to build a system that can look at a person's hand through a
webcam and correctly guess which Namibian Sign Language sign they are
showing, starting with a small set of signs like greetings and letters.

**Measurable**
Success is measured by how many signs the model gets right out of the
ones I trained it on. When I run `train_model.py`, it prints an accuracy
report for each sign, so I can see exactly which ones the model is
struggling with and go collect more samples for those.

**Achievable**
Instead of trying to train a deep learning model from raw video, which
would need a lot of data and a strong GPU, I used MediaPipe to just track
21 points on each hand and fed those points into a RandomForest
classifier. This keeps things realistic for a student laptop and a small
amount of self collected data.

**Relevant**
There isn't a well known NSL recognition tool out there, and most sign
language projects online are built for ASL. This project tries to close
that gap a little, even if it starts small, and it also gave me a chance
to practice computer vision and machine learning basics in a project that
actually means something to my own context.

**Time bound**
This is being built in stages. First stage is getting single hand shape
signs working properly (the current version of the project). Later
stages, if I keep working on it, would be adding signs that involve
motion, and maybe collecting help from actual Deaf signers to check that
the signs I am recording are correct.

## How it actually works

1. When you run the webcam scripts, MediaPipe's hand landmark model looks
   at each frame and finds 21 points on your hand (fingertips, knuckles,
   wrist, and so on). The first time you run it, it downloads a small
   model file for this, so you need internet just for that one time.
2. Those 21 points get saved into a CSV file along with whatever label
   you typed in (like HELLO or THANK_YOU). This is the training data.
3. `train_model.py` reads that CSV and trains a RandomForest model to
   learn which point patterns match which sign.
4. `translate_live.py` runs the same point detection live, feeds it into
   the trained model, and shows you the predicted sign on screen. It also
   waits for a few frames of the same prediction before adding it to a
   sentence, so it does not just flicker between random guesses.

## Setup

I used VS Code and Spyder to build this, so here is how to set it up in
either one.

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

In Spyder, open the folder as a project and run each script from the run
button. In VS Code, open the folder, pick the venv as your interpreter in
the bottom right corner, and run scripts either with the run arrow or by
typing `python filename.py` in the terminal.

## How to use it, step by step

### Step 1: Collect samples for a sign
```bash
python collect_data.py
```
It will ask you to type a label for the sign, like HELLO. Hold the sign
in front of your camera and press `s` a bunch of times to save samples,
moving your hand slightly each time so the model sees some variation.
I aimed for around 150 to 300 samples per sign. Press `q` when done.

### Step 2: Repeat for every sign you want
Run `collect_data.py` again for each new sign with a different label.
Everything gets added to the same `data/landmarks.csv` file, so you do
not lose earlier signs when you add new ones.

### Step 3: Train the model
```bash
python train_model.py
```
This trains the model on everything in your CSV and prints out how well
it did per sign. If one sign has low accuracy, go back and record more
samples for just that one.

### Step 4: Run it live
```bash
python translate_live.py
```
Show your webcam a sign you trained and it should show the predicted
sign and confidence on screen, and build a sentence as you go. Press `c`
to clear the sentence and `q` to quit.

## Things I kept in mind for the Namibian context

- Namibia does not have one single, well documented, standard sign
  language dataset, so I am not assuming my recorded signs are perfect.
  If I want this to actually be useful, I would need to check my signs
  with real Deaf Namibian signers, maybe through an organization like the
  Namibia National Association of the Deaf (NNAD), instead of guessing
  based on ASL or SASL videos.
- Everything runs fully offline once it is set up. No cloud calls needed
  after the first run. This matters because data can be expensive and
  internet is not always reliable everywhere in Namibia.
- The whole thing is built to run on a normal laptop without a GPU, since
  that is what most students realistically have access to.

## What this project does not do yet

- It only recognizes signs that are a single still hand shape. It cannot
  yet handle signs that involve movement, since that would need a
  different kind of model that looks at sequences of frames instead of
  just one frame at a time.
- Accuracy really depends on how much data I collect and how varied it
  is. A sign trained with only 20 samples from one angle will not work as
  well as one trained with 200 samples from different angles and
  distances.
- Lighting and background can affect accuracy if they are very different
  between when I collected the data and when I actually demo it, so I try
  to test in similar conditions to where I plan to use it.
