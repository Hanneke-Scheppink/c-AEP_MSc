% Reset workspace
clc;
clear variables;
close all;

% Add Matlab Noise-Tagging toolbox
addpath(genpath('/users/u829215/mant'));

% Codes
src_codes_dir = '/Users/u829215/mant/mant/codes';
tmp = load(fullfile(src_codes_dir, 'mgold_61_6521.mat'));
codes = tmp.codes(:, 1:end-2);  % remove m-sequences
fprintf('codes %d x %d\n', size(codes, 1), size(codes, 2));

% Criterium 1.1: start with a 1
idx1 = codes(1, :) == 1;
fprintf('C1.1: start with a 1: %d/%d\n', sum(idx1), numel(idx1));
codes = codes(:, idx1);

% Criterium 1.2: end with a 0
idx2 = codes(end, :) == 0;
fprintf('C1.2: end with a 0: %d/%d\n', sum(idx2), numel(idx2));
codes = codes(:, idx2);

% Criterium 1.3: balanced event distribution (#1 ~ #2)
[E, e] = jt_reconvolution_event_matrix(codes, 'duration');
event_distribution = squeeze(sum(E, 1));
val = min(abs(event_distribution(1, :) - event_distribution(2, :)));
idx3 = abs(event_distribution(1, :) - event_distribution(2, :)) == val;
fprintf('C1.3: codes with balanced events: %d/%d (1=%d and 2=%d)\n', ...
    sum(idx3), numel(idx3), ...
    event_distribution(1, find(idx3)), event_distribution(2, find(idx3)));
code = codes(:, idx3);

% Selected code
index = 1:size(tmp.codes, 2) - 2;
index = index(idx1);
index = index(idx2);
index = index(idx3);
fprintf('Selected code: %d\n', index);

% Plot codes
figure();
hold on;
index = 1:size(tmp.codes, 2) - 2;
for i = 1:numel(index)
    plot(1:size(tmp.codes, 1), index(i) + 0.9*tmp.codes(:, index(i)), '-k');
end
index = index(idx1);
for i = 1:numel(index)
    plot(1:size(tmp.codes, 1), index(i) + 0.9*tmp.codes(:, index(i)), '-b');
end
index = index(idx2);
for i = 1:numel(index)
    plot(1:size(tmp.codes, 1), index(i) + 0.9*tmp.codes(:, index(i)), '-g');
end
index = index(idx3);
for i = 1:numel(index)
    plot(1:size(tmp.codes, 1), index(i) + 0.9*tmp.codes(:, index(i)), '-r');
end

% Criterium 2.1: minimum auto-correlation
auto_correlation = squeeze(jt_correlation(code, code, struct('method', 'shift')));
val = min(abs(auto_correlation));
idx1 = abs(auto_correlation) == val;
fprintf('C2.1: minimum auto-correlation: %d/%d\n', sum(idx1), numel(idx1));

% Criterium 2.2: largest delay relative to original 
% Note, maintaining C1.1, C1.2, etc., so performed manually step by step 
lag = [0:1:numel(code)/2-1 numel(code)/2-1:-1:0];
val = max(lag(idx1));
idx2 = lag(idx1) == val;
fprintf('C2.2: "minimum" lag: %d/%d\n', sum(idx2), numel(idx2));

% Selected lag
lag = 61;
fprintf('Selected lag: %d\n', lag);

% Plot lag
figure();
hold on;
lags = 1:size(codes, 1);
plot(1:numel(auto_correlation), auto_correlation, '-k');
plot(lags(idx1), auto_correlation(idx1), '.b');
plot(lag, auto_correlation(lag), '.g');
plot(lag(1), auto_correlation(lag(1)), '.r');
xlabel('lag [sample]');
ylabel('correlation');
legend({'auto-correlation', 'C2.1', 'C2.2', 'selected'});

% Put it all together
src_codes_dir = '/Users/u829215/mant/mant/codes';
tmp = load(fullfile(src_codes_dir, 'mgold_61_6521.mat'));
code1 = tmp.codes(:, 28);  % code id 28
code2 = circshift(code1, 61);  % lag id 61

figure();
hold on;
plot(jt_upsample(code1, 20, 1));
plot(-1.5 + jt_upsample(code2, 20, 1));
legend({'code1', 'code2'});

[E, e] = jt_reconvolution_event_matrix(cat(2, code1, code2), 'duration');
event_distribution = squeeze(sum(E, 1));
fprintf('code1: 1=%d, 2=%d\n', event_distribution(1, 1), event_distribution(2, 1));
fprintf('code2: 1=%d, 2=%d\n', event_distribution(1, 2), event_distribution(2, 2));

rho = jt_correlation(code1, code2);
fprintf('rho(code1, code2) = %.4f (= 1/(2^6-1))\n', rho);
