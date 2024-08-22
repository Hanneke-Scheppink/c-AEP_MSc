#!/home/artiintel/hanschep/.conda/envs/env_hs/bin/python
# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import pyntbci
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LogReg

# Get params from environment
n_comps = int(os.getenv("n_comps", "1"))
gate = os.getenv("gate", "None")
cca_meth = os.getenv("cca_meth", "ecca")
condition = os.getenv("cond", "unmodulated")
envelope_method = os.getenv("envelope_method", "filterbank")
fs = int(os.getenv("fs", "120"))

encoding_lag = None if os.getenv("encoding", "None") == "None" else float(os.getenv("encoding", "None"))
decoding_shift = None if os.getenv("decoding", "None") == "None" else float(os.getenv("decoding", "None"))
sec_length = float(os.getenv("secL", "5"))

# Define parameters
n_folds = 4
decision_window_lengths = [1, 2, 5, 10, 20, 30]
decision_window_stride = 0.1


class GatingComps():
    def __init__(self):
        self.a = 0

    def decision_function(self, X):
        """
        Returns the component scores X; needed as the pyntbci package could not return all components with
        the predefined gates.

        X: np.ndarray Score matrix of shape (n_trials, n_classes, n_components)

        Returns:
        scores: np.ndarray Score matrix of shape (n_trials, n_classes, n_components)
        """
        return X

    def fit(self, X, y):
        b = 0


def e_r_cca_short_windows(x, e_l, e_r, v, sec, fs, e_r_cca, gate_=None):
    """
    Returns the predicted labels for each moving window (here of length tau) over test data x

    x: np.ndarray; data matrix of shape (1, channels, samples)
    e_l: np.array; envelope (left) matrix of shape (samples)
    e_r: np.array; envelope (right) matrix of shape (samples)
    v: None or np.ndarray; noise-code matrix of shape (classes, samples)
    sec: int; length of decision window
    fs: int; sampling frequency
    e_r_cca: list[TransformerMixin]; the CCA (either eCCA or rCCA) that was fit on the training data
    gate: None or BaseEstimator; classifier used to combine the scores obtained from the CCA components (here LDA or LogReg)

    Returns:
    results: np.ndarray; array with predicted labels for each of the moving windows of length sec
    """
    results = []
    window_shift_size = (1 / int(fs / 2)) * fs
    range_len = int((x.shape[2] / fs - sec) / window_shift_size) + 1
    for j in range(0, range_len):
        start = int(j * window_shift_size * fs)
        end = int(start + (sec * fs))
        stim_ = np.concatenate((np.expand_dims(e_l[start:end], axis=0), np.expand_dims(e_r[start:end], axis=0)), axis=0)
        if v is None:
            e_r_cca.set_stimulus(stim_)
        else:
            e_r_cca.set_stimulus_amplitudes(stimulus=v[:, start:end], amplitudes=stim_)
        if gate_ is None:
            y_hat = e_r_cca.predict(x[:, :, start:end])
        else:
            tst_scores = e_r_cca.decision_function(x[:, :, start:end])
            features_tst = np.array(tst_scores[0][0] - tst_scores[0][1]).reshape(1, -1)
            y_hat = gate_.predict(features_tst)
        results.append(y_hat[0])
    return np.array(results)


def e_r_cca_decoding(Xx, y, Ee, Vv, n_folds, decision_window_lengths, fs, gate, encoding_lag, sec_length, decoding_shift,
                     n_comps):
    # Please note; the envelopes and EEG are cut such that the folds can be equally made
    # Furthermore, not that no y_trn and y_tst are made, as the folds are created in each trial, for which the y remains the same
    cutting_shape = int(Ee.shape[2] / n_folds) * n_folds
    X = Xx[:, :, :cutting_shape]
    E = Ee[:, :, :cutting_shape]
    if Vv is not None:
        V = np.tile(Vv, (1, 1, int(np.ceil(X.shape[2] / Vv.shape[2]))))[:, :, :X.shape[2]]

    # Create folds
    folds = np.arange(n_folds).repeat(int(X.shape[2] / n_folds))
    accuracy = np.empty([X.shape[0], n_folds, len(decision_window_lengths)])

    # Loop folds
    for i_fold in range(n_folds):
        # Split data to train and test set
        X_trn = X[:, :, folds != i_fold]
        E_trn = np.empty([X_trn.shape[0], X_trn.shape[2]])
        if Vv is None:
            V_trn = None
        else:
            V_trn = np.empty([X_trn.shape[0], X_trn.shape[2]])
            V_tst = V[:, :, folds == i_fold]
            V_trn_lda = V[:, :, folds != i_fold]
        for trial in range(X.shape[0]):
            E_trn[trial, :] = E[trial, y[trial], folds != i_fold]
            if V_trn is None:
                V_trn = None
            else:
                V_trn[trial, :] = V[trial, y[trial], folds != i_fold]
        E_trn_lda = E[:, :, folds != i_fold]
        X_tst = X[:, :, folds == i_fold]
        E_tst = E[:, :, folds == i_fold]
        stims_att = np.arange(E_trn.shape[0])

        if gate == 'LDA':
            gating = GatingComps()
            gate_ = LDA()
        elif gate == 'None':
            gating = None
            gate_ = None
        elif gate == 'LogReg':
            gating = GatingComps()
            gate_ = LogReg()
        else:
            raise Exception(f"The gating you defined is not known, please try None, LDA or LogReg; you inserted: {gate}")

        if Vv is None:
            # No noise-codes are used, so use eCCA
            V_inter = Vv
            cca = pyntbci.classifiers.rCCA(stimulus=E_trn, fs=fs, event="id", onset_event=False,
                                           decoding_length=decoding_shift, encoding_length=encoding_lag,
                                           n_components=n_comps, gating=gating)
        else:
            # Noise-codes are employed so use rCCA
            cca = pyntbci.classifiers.rCCA(stimulus=V_trn, fs=fs, event="contrast", onset_event=False,
                                           decoding_length=decoding_shift, encoding_length=encoding_lag,
                                           amplitudes=E_trn, n_components=n_comps,
                                           gating=gating)
        cca.fit(X_trn, stims_att)

        if gate == "LogReg" or gate == "LDA":
            range_len_lda = int((X_trn[[0], :, :].shape[2] / fs - sec_length) / sec_length) + 1
            y_lda = np.ones([X_trn.shape[0], range_len_lda])
            feature_l = np.empty([X_trn.shape[0], range_len_lda, n_comps], dtype='object')
            for t in range(X_trn.shape[0]):
                y_lda[t, :] = list(np.repeat([y[t]], range_len_lda))
                for l in range(0, range_len_lda):
                    start = int(l * sec_length * fs)
                    end = int(start + (sec_length * fs))
                    if Vv is None:
                        cca.set_stimulus(E_trn_lda[t, :, start:end])
                    else:
                        cca.set_stimulus_amplitudes(stimulus=V_trn_lda[t, :, start:end],
                                                    amplitudes=E_trn_lda[t, :, start:end])
                    scores_l_r = cca.decision_function(X_trn[[t], :, start:end])  # X: n_trs, n_ch, n_sam
                    feature_l[t, l, :] = scores_l_r[0][0] - scores_l_r[0][1]
            feature_trn = feature_l.reshape(feature_l.shape[0] * feature_l.shape[1], n_comps)
            gate_.fit(feature_trn, y_lda.flatten())

        for t in range(X_trn.shape[0]):
            for idx_sec, second in enumerate(decision_window_lengths):
                if Vv is None:
                    results = e_r_cca_short_windows(X_tst[[t], :, :], E_tst[t, 0, :], E_tst[t, 1, :], None, second,
                                                   fs, cca, gate_)
                else:
                    results = e_r_cca_short_windows(X_tst[[t], :, :], E_tst[t, 0, :], E_tst[t, 1, :], V_tst[t, :, :],
                                                   second,
                                                   fs, cca, gate_)
                acc = np.count_nonzero(results == y[t]) / len(results)
                accuracy[t, i_fold, idx_sec] = acc
    return accuracy.mean(axis=(0, 1))


derivatives = os.path.join(os.path.expanduser("~"),  "Documents", "MScThesis", "caep", "experiment", "parallel", "derivatives")
with open(os.path.join(derivatives, f"E_{envelope_method}.pickle"), 'rb') as f:
    E = pickle.load(f)
with open(os.path.join(derivatives, f"X.pickle"), 'rb') as f:
    X = pickle.load(f)
# For completeness, due to problems the loaded y will not be used for the first participant
with open(os.path.join(derivatives, f"y.pickle"), 'rb') as f:
    y_load = pickle.load(f)
y = np.array([0, 0, 1, 0, 0, 1, 0, 1])

if cca_meth == "rcca":
    with open(os.path.join(derivatives, f"V.pickle"), 'rb') as f:
        V_load = pickle.load(f)
    V = np.tile(V_load['modulated'], (X['modulated'].shape[0], 1, 1))
else:
    V = None

accuracies = e_r_cca_decoding(X[condition], y, E[condition], V, n_folds, decision_window_lengths, fs, gate,
                              encoding_lag, sec_length, decoding_shift, n_comps)

with open(os.path.join(derivatives, "derivatives",
                       f"{cca_meth}_{condition}_envmeth{envelope_method}_accuracies_Gate{gate}_Enc{encoding_lag}_Dec{decoding_shift}_secL{int(sec_length)}_Comp{n_comps}.pickle"),
          'wb') as f:
    pickle.dump(accuracies, f, protocol=pickle.HIGHEST_PROTOCOL)

