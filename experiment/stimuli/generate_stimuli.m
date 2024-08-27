% Reset workspace
clc;
clear variables;
close all;

% Directories
src_audio_dir = '~/caep/experiment/parallel/stimuli/original';
dst_audio_dir = '~/caep/experiment/parallel/stimuli';
src_codes_dir = '~/caep/experiment/parallel/stimuli';

% Audio (carrier)
n_tracks = 8;
n_parts = 4;
target_loudness = -23;  % dB

% Noise-codes (modulator)
tmp = load(fullfile(src_codes_dir, 'mgold_61_6521.mat'));
codes = tmp.codes(:, [28, 28]);  % code id 28
codes(:, 2) = circshift(codes(:, 2), 61);  % lag id 61
n_codes = size(codes, 2);

% Parameters to modulate the audio file
bitrate = 40;  % Hz
moddepth = 0.7;  % 0-1
smooth_value = 1;  % 1 >

% Make output folder
if ~exist(dst_audio_dir, 'dir')
    mkdir(dst_audio_dir)
end

% Modulate audio
for i_track = 1:n_tracks
    for i_part = 1:n_parts
        fn = sprintf('t%d_p%d.wav', i_track, i_part);
        fprintf('%s\n', fn)

        % Read original
        [audio, fs] = audioread(fullfile(src_audio_dir, fn)); 
        audio = audio(:, 1);  % make mono
    
        % Adjust loudness
        loudness = integratedLoudness(audio, fs);
        gain = 10^((target_loudness - loudness) / 20);
        audio = audio .* gain;
        
        % Save original
        audiowrite(fullfile(dst_audio_dir, fn), audio, fs);
        
        for i_code = 1:n_codes
            fn = sprintf('t%d_p%d_c%d.wav', i_track, i_part, i_code-1);
    
            % Modulate audio with code
            [modulated_audio, ~, ~, ~] = modulate_carrier(audio, codes(:, i_code), fs, bitrate, moddepth, smooth_value);

            % Save modulated
            audiowrite(fullfile(dst_audio_dir, fn), modulated_audio, fs);
        end
    end
end
