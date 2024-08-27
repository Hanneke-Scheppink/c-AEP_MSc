# To install:
# mkvirtualenv -p=3.11 pyscab
# pip install D:\scab-python-master\PyAudio-0.2.11-cp38-cp38-win_amd64.whl
# pip install pyscab pylsl rusocsci pandas
# pip install psychopy
# pip install python-bidi==0.4.2
#
# To run:
# D:
# cd Users\bci\caep\experiment\parallel
# workon pyscab
# python caep_parallel.py

import os
import glob
import json
import sys
import time

import pyscab
import pandas as pd
from pylsl import StreamInfo, StreamOutlet
import psychopy
from psychopy import visual, core, event, monitors, misc
from rusocsci import buttonbox


def send_marker(val):
    outlet.push_sample([json.dumps(json.dumps({"start_audio": val}))])
    bb.sendMarker([val])  # set
    time.sleep(0.002)  # required for BrainAmp
    bb.sendMarker([0])  # reset
    print(f"\tmarker sent: {val}")


subject = 1
audio_dir = r"D:\Users\bci\caep\experiment\parallel\stimuli"
design_dir = r"D:\Users\bci\caep\experiment\parallel\designs"
question_dir = r"D:\Users\bci\caep\experiment\parallel\questions"
start_time = 5.0
cue_time = 1.0
blank_time = 1.0
iti_time = 1.0
feedback_time = 1.0
n_trials_break = 2
screen_id = 0
screen_width_pix = 1920
screen_height_pix = 1080
screen_width_cm = 53.0
screen_distance_cm = 60.0
screen_color = (0, 0, 0)
fixation_size = 0.5
fixation_color = (-1, -1, -1)
text_height = 1.0
text_color = (-1, -1, -1)
question_keys = ["z", "x", "c", "v"]

# Setup monitor
monitor = monitors.Monitor(name="MM00.440")
monitor.setSizePix((screen_width_pix, screen_height_pix)) 
monitor.setWidth(screen_width_cm)
monitor.setDistance(screen_distance_cm)
monitor.saveMon()
ppd = misc.deg2pix(1.0, monitor)

# Setup window
window = visual.Window(monitor=monitor, screen=screen_id, units="pix", size=(screen_width_pix, screen_height_pix),
                       color=screen_color, fullscr=True, waitBlanking=False, allowGUI=False)

# Setup mouse
mouse = event.Mouse(win=window, newPos=(-2000, 2000), visible=False)

# Setup fixation cross for visual task
fixation = visual.TextStim(win=window, units="pix", pos=(0, 0), height=int(fixation_size * ppd), text="+",
                           color=fixation_color, autoDraw=True)

# Setup text box for instructions and cueing
text = visual.TextStim(win=window, units="pix", pos=(0, int(2 * ppd)), height=int(text_height * ppd), text="",
                       color=text_color, wrapWidth=screen_width_pix, autoDraw=True)

# Setup buttonbox for hardware markers
bb = buttonbox.Buttonbox()

# Setup LSL for software markers
outlet = StreamOutlet(StreamInfo(name="MarkerStream", type="Markers", channel_count=1, nominal_srate=0,
                                 channel_format="string", source_id="MarkerStream"))

# Load design
fn = os.path.join(design_dir, f"sub-{subject:02d}.csv")
design = pd.read_csv(fn, index_col=0)

# Load questions
fn = os.path.join(question_dir, "questions.csv")
questions = pd.read_csv(fn, sep=";")

# Load audio
afh = pyscab.DataHandler(frame_rate=44100)
audio_mapping = {}
for i_file, file in enumerate(glob.glob(os.path.join(audio_dir, "*.wav"))):
    file_id = 1 + i_file
    afh.load(file_id, os.path.join(audio_dir, file), volume=1.0)
    audio_mapping[os.path.basename(file)[:-4]] = file_id

# Setup audio I/F
ahc = pyscab.AudioInterface(device_name="X-AIR ASIO Driver", n_ch=2, format="INT16", frames_per_buffer=512,
                            frame_rate=44100)

# Setup stimulation protocol
share = [0 for m in range(8)]
stc = pyscab.StimulationController(ahc, marker_send=send_marker, share=share, correct_latency=True,
                                   correct_hardware_buffer=True)
stc.open()

# Present trials
for i_trial in range(len(design)):

    # Break
    if i_trial % n_trials_break == 0:
        text.text = "Waiting for researcher to continue"
        print("Waiting for researcher to continue")
        window.callOnFlip(outlet.push_sample, [json.dumps("waiting")])
        window.flip()
        event.waitKeys(keyList=["c"])

        outlet.push_sample([json.dumps(json.dumps({
            "python_version": sys.version, 
            "psychopy_version": psychopy.__version__}))])
        outlet.push_sample([json.dumps(json.dumps({"design_file": fn}))])
        outlet.push_sample([json.dumps(json.dumps({"questions_file": fn}))])

        # Continue
        text.text = "Starting"
        print("Start run")
        window.callOnFlip(outlet.push_sample, [json.dumps("start_run")])
        window.flip()
        core.wait(start_time)

    trial = 1 + i_trial
    print(f"trial {trial:d}/{len(design):d}")
    outlet.push_sample([json.dumps({"start_trial": 1 + i_trial})])

    # Present cue
    cue = design.iloc[i_trial]["attention"]
    text.text = cue
    print(f"\tcue: {cue}")
    window.callOnFlip(outlet.push_sample, [json.dumps({"start_cue": cue})])
    window.flip()
    core.wait(cue_time)
    
    # Blank
    text.text = ""
    print("\tblank")
    window.callOnFlip(outlet.push_sample, [json.dumps("start_blank")])
    window.flip()
    core.wait(blank_time)

    # Present audio
    audio_left = design.iloc[i_trial]["left"]
    audio_right = design.iloc[i_trial]["right"]
    print(f"\taudio: {audio_left} {audio_right}")
    window.callOnFlip(outlet.push_sample, [json.dumps({"audio_left": audio_left, "audio_right": audio_right})])
    stc.play([
        [0.0, audio_mapping[audio_left], [1], 1],  # [time, file_id, ch, marker]
        [0.0, audio_mapping[audio_right], [2], 2]],  # [time, file_id, ch, marker]
        afh)
    
    # Inter-trial interval
    text.text = ""
    print("\tinter-trial interval")
    window.callOnFlip(outlet.push_sample, [json.dumps("start_iti")])
    window.flip()
    core.wait(iti_time)
    
    if i_trial % 2 == 1:
        for i_question in [-1, 0]:
    
            # Present question
            question = questions.iloc[i_trial + i_question]["question"]
            correct = questions.iloc[i_trial + i_question]["correct"]
            answers = [f"({z}) " + questions.iloc[i_trial + i_question][f"answer_{z}"] for z in "abcd"]
            text.text = question + "\n\n" + "\n".join(answers)
            print(f"\tQuestion: {question}")
            window.callOnFlip(outlet.push_sample, [json.dumps({"start_question": question, "answers": answers, "correct": correct})])
            window.flip()

            # Wait response
            keys = event.waitKeys(keyList=question_keys)
            response = keys[0]
            outlet.push_sample([json.dumps({"start_response": response})])
            print(f"\tResponse: {response}")

            # Present feedback
            if question_keys.index(response) == "abcd".index(correct):
                feedback = "Correct, well done!"
            else:
                feedback = "Incorrect, keep on trying!"
            print(f"\tFeedback: {feedback})")
            text.text = feedback
            window.callOnFlip(outlet.push_sample, [json.dumps({"start_feedback": feedback})])
            window.flip()
            core.wait(feedback_time)

# Wait for researcher
print("Waiting for researcher to stop")
text.text = "Waiting for researcher to stop"
window.callOnFlip(outlet.push_sample, [json.dumps("stop_experiment")])
window.flip()
event.waitKeys()

# Close everything
print("Quit")
mouse.setVisible(True)
window.close()
stc.close()
core.quit()
