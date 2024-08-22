"""
This file is used to load and pre process the EEG data. Furthermore, the codes (V) are loaded and the two optimal codes selected.
The audio files are loaded, and the envelopes created according to the defined envelope method (filterbank, rms, hilbert)
The labels y are loaded according to the saved parameters from the experiment, were attention to the "left" is denoted with label 0, and "right" with label 1

The EEG data are stored per condition (modulated, unmodulated), as a matrix in shape [trials, channels, samples]
The envelopes are stored per condition, as a matrix in shape [trials, classes, samples], hence the envelope from the left and right side of a trial is combined
The codes are stored for the modulated condition, as a matrix in shape [classes, samples]

"""
# Imports
import os
import scipy.io as sio
import numpy as np
import pyxdf
import mne
from mnelab_read_raw import read_raw_xdf as read_raw
import json
import scipy.signal as sps
import pickle
import pyntbci


# Path definition:
datapath = os.path.join(os.path.expanduser("~"),  "Documents", "MScThesis", "caep", "experiment", "parallel", "sourcedata")
stim_fol = os.path.join(os.path.expanduser("~"),  "Documents", "MScThesis", "caep", "experiment", "parallel", "stimuli")


# Load the subject and run info from the environment
subject = os.getenv("subject", "P001")
session = os.getenv("session", "S001")
nmb_runs = int(os.getenv("nmb_runs", "8"))
envelope_method = os.getenv("envelope_method", "filterbank")
fs = int(os.getenv("fs", "120"))

# Define Parameters
notch = 50
bandpass = (1, 20)
intermediate_fs_audio = 8000
lowpass_value = bandpass[1]
t_max = 2.66 * 60 # to make all trials equally long

# Build a bandpass filter (here as a lowpass for the Hilbert transform)
N_bp, Wn_bp = sps.buttord(wp=lowpass_value - 0.45, ws=lowpass_value + 0.45, gpass=0.5, gstop=15,fs=intermediate_fs_audio)
bbutter, abutter = sps.butter(N=N_bp, Wn=Wn_bp, btype="low", fs=intermediate_fs_audio)

# Define dictionaries for the data
X = dict()
y = dict()
V = dict()
E = dict()


# Load codes
code_id = 27
code_shift = 61
codes = sio.loadmat(os.path.join(stim_fol, "stimuli", "mgold_61_6521.mat"))["codes"].T

# Select codes
codes = codes[[code_id, code_id], :]

# Shift codes
codes[1, :] = np.roll(codes[1, :], code_shift)

V['modulated'] = np.repeat(codes, int(fs / 40), axis=1)


# Read EEG
def load_eeg(fn, sub, run, fs_target, notch, bandpass):
    streams = pyxdf.resolve_streams(fn)
    names = [stream["name"] for stream in streams]
    stream_id = streams[names.index("BrainAmpSeries-Dev_1")]["stream_id"]
    raw = read_raw(fn, stream_ids=[stream_id])
    t_max = 2.66 * 60

    events = mne.find_events(raw, stim_channel="triggerStream", verbose=False)
    events = events[::2, :]  # marker triggered by both audio at left and right, skip every second
    raw.drop_channels("triggerStream")

    raw.notch_filter(freqs=np.arange(notch, raw.info["sfreq"] / 2, notch), picks="eeg", verbose=False)
    raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], picks="eeg", verbose=False)
    print("len raw:", raw.__len__())
    epo = mne.Epochs(raw, events=events, tmin=-0.5, tmax=0.5 + t_max, baseline=None, picks="eeg", preload=True,
                     verbose=False)

    epo.resample(sfreq=fs_target, verbose=False)

    X = epo.get_data(tmin=0, tmax=t_max, copy=True)
    return X

# Define Hilbert transform
def compute_hilbert_envelope(audio_signal, aFs, fs_inter, target_fs, low_pass, bbutter = bbutter):
    signal_hilb = sps.hilbert(audio_signal)
    amplitude_envelope = np.abs(signal_hilb)
    if low_pass is None:
        env =  sps.resample(amplitude_envelope, int(amplitude_envelope.size/aFs) * target_fs)
    else:
        resam_env = sps.resample(amplitude_envelope, int(amplitude_envelope.size / aFs) * fs_inter)
        lp_env = sps.filtfilt(bbutter, 1, resam_env)
        env =  sps.resample(lp_env, int(lp_env.size/fs_inter) * target_fs)
    return env


# Load the envelope
def envelope_load(env_method, audio_signal, aFs, fs_inter, target_fs, bbutter, low_pass):
    if env_method == "hilbert":
        env = compute_hilbert_envelope(audio_signal, aFs, fs_inter, target_fs, low_pass, bbutter)
    elif env_method == "filterbank":
        tmp = pyntbci.envelope.envelope_gammatone(audio=audio_signal, fs=aFs, fs_inter=fs_inter,fs_target=target_fs, power=0.6, lowpass=low_pass)
        env = tmp.sum(axis=1)
    elif env_method == "rms":
        env = pyntbci.envelope.envelope_rms(audio=audio_signal, fs=aFs, fs_inter=fs_inter, fs_target=target_fs)
    else:
        raise Exception(f"The envelope you defined is not known, please try hilbert, filterbank or rms; you inserted: {env_method}")
    return env


for run in range(1, nmb_runs + 1):
    fn = os.path.join(datapath, f"sub-{subject}", f"ses-{session}", "eeg",
                      f"sub-{subject}_ses-{session}_task-audio_run-00{run}_eeg.xdf")
    streams = pyxdf.load_xdf(fn)[0]
    names = [stream["info"]["name"][0] for stream in streams]
    marker_stream = streams[names.index("MarkerStream")]

    X_ = load_eeg(fn, subject, run, fs, notch, bandpass)

    cues = [json.loads(marker[0])["start_cue"]
            for marker in marker_stream["time_series"]
            if isinstance(json.loads(marker[0]), dict) and "start_cue" in json.loads(marker[0])]

    audio_left = [json.loads(marker[0])["audio_left"]
                  for marker in marker_stream["time_series"]
                  if isinstance(json.loads(marker[0]), dict) and "audio_left" in json.loads(marker[0])]

    audio_right = [json.loads(marker[0])["audio_right"]
                   for marker in marker_stream["time_series"]
                   if isinstance(json.loads(marker[0]), dict) and "audio_right" in json.loads(marker[0])]

    for trial in range(len(audio_left)):
        aFsL, audio_L = sio.wavfile.read(f'{os.path.join(stim_fol, audio_left[trial])}.wav')
        aFsR, audio_R = sio.wavfile.read(f'{os.path.join(stim_fol, audio_right[trial])}.wav')
        E_L = envelope_load(envelope_method, audio_L, aFsL, intermediate_fs_audio, fs, bbutter_hil, bandpass[1])[:int(t_max * fs)]
        E_R = envelope_load(envelope_method, audio_R, aFsR, intermediate_fs_audio, fs, bbutter_hil, bandpass[1])[:int(t_max * fs)]
        E_ = np.expand_dims(np.concatenate((np.expand_dims(E_L / E_L.max(), axis=0),
                                            np.expand_dims(E_R / E_R.max(), axis=0)), axis=0), axis=0)

        y_ = np.array([0]) if cues[trial] == 'left' else np.array([1])

        if len(audio_left[trial]) > 5:
            condition = 'modulated'
        else:
            condition = 'unmodulated'

        if condition in X:
            X[condition] = np.concatenate((X[condition], X_[[trial], :, :]), axis=0)
            E[condition] = np.concatenate((E[condition], E_), axis=0)
            y[condition] = np.concatenate((y[condition], y_))
        else:
            X[condition] = X_[[trial], :, :]
            E[condition] = E_
            y[condition] = y_


derivatives = os.path.join(os.path.expanduser("~"),  "Documents", "MScThesis", "caep", "experiment", "parallel", "derivatives")
with open(os.path.join(derivatives, f"X_{fs}.pickle"), 'wb') as f:
    pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(derivatives, f"E_{fs}.pickle"), 'wb') as f:
    pickle.dump(E, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(derivatives, f"y_{fs}.pickle"), 'wb') as f:
    pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(derivatives, f"V_{fs}.pickle"), 'wb') as f:
    pickle.dump(V, f, protocol=pickle.HIGHEST_PROTOCOL)










