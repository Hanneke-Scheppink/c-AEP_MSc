function [modulated_carier, scaled_modulator, smooth_modulator, modulator] = modulate_carrier(carrier, modulator, fs, bitrate, moddepth, smoothval)
% [modulated_carier, scaled_modulator, smooth_modulator, modulator] = modulate_carrier(carier, modulator, fs, bitrate, moddepth, smoothval)
% Modulates a carrier with a modulator. In order to have a soft sound, the 
% edges of the (binary) modulator are smoothed using a sine wave.
%
% Input:
%   carrier: [n_carrier 1]
%       The original audio time-series.
%   modulator: [n_modulator 1]
%       The original noise-code time-series.
%   fs: int
%       The sampling frequency of the carier.
%   bitrate: int
%       The sampling frequency (i.e., presentation rate) of the modulator.
%   moddepth: float
%       The modulation depth of the modulator, between 0 and 1. 
%   smoothval: float (default 1.0)
%       The smoothness of the smoothing. Lower is smoother. Value > 1.

if nargin < 6 || isempty(smoothval); smoothval = 1.0; end

n_carrier = numel(carrier);
n_modulator = numel(modulator);

% Upsample code to audio fs given the code bitrate
uprate = ceil(fs / bitrate);
modulator = double(reshape(repmat(modulator, [1 uprate])', [], 1));

% Repeat code to audio length
n_reps = ceil(n_carrier / (uprate * n_modulator));
modulator = repmat(modulator, n_reps, 1);
modulator = modulator(1:n_carrier);
    
% Create the smooth functions (left and right) for the code
if mod(uprate, 2) == 0
    smooth_fn_r = zeros(uprate, 1);
    t = -pi / (smoothval * 2) : pi / (uprate - 1) : pi / (smoothval * 2);
else 
    smooth_fn_r = zeros(uprate + 1, 1);
    t = -pi / (smoothval * 2) : pi / (uprate) : pi / (smoothval * 2);
end
y = (sin(smoothval * t) + 1) / 2;
n_y = length(y);
h = round((uprate - n_y) / 2);
if smoothval == 1
    smooth_fn_r = y;
else
    smooth_fn_r(h : (h + n_y - 1)) = y;
    smooth_fn_r((h + n_y) : end) = 1;
end
smooth_fn_l = smooth_fn_r(end:-1:1);

% Make the code edges smooth
smooth_modulator = modulator;
df = find(abs(diff(modulator)) == 1);
rn = round(uprate / 2);
for i = 1:numel(df)
    if modulator(df(i)) == 1
        smooth_modulator(df(i) - rn + 1 : df(i) + rn) = smooth_fn_l;
    else
        smooth_modulator(df(i) - rn + 1 : df(i) + rn) = smooth_fn_r;
    end
end
smooth_modulator = smooth_modulator(1:n_carrier);

% Scale the smoothed code with the modulation depth
scaled_modulator = (1 - moddepth) .* (1 - smooth_modulator) + smooth_modulator;

% Modulate the audio
modulated_carier = scaled_modulator .* carrier;
