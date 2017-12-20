function [speech_hat, noise_hat, cost_noisy] = MRIdenoising(noisy_sig, noise_est, fs, config)
% MRIdenoising Denoise speech corrupted by MRI noise.
% Remove MRI noise from a speech recording using complex non-negative matrix
% factorization with intra-source additivity (CMF-WISA) and additional
% frequency and temporal regularization terms.
%
% Inputs:
%   noisy_sig: [vector]
%       noisy speech
%   noise_est: [vector] or [scalar]
%       [vector]: samples of the noise in the noisy speech
%       [scalar]: number of seconds at the beginning of noisy_sig to use to
%       estimate the noise. 0 means to automatically determine the number
%       seconds (see Automatic Speech Start Detection Parameters below).
%   fs: [scalar]
%       sampling rate of the speech
%   config: [structure] (optional)
%       configuration parameters
%       -------------------------------
%       Matrix Factorization Parameters
%       -------------------------------
%       config.nfft: [scalar] (default: 1024)
%           nfft-point FFT used for calculating the spectrogram 
%       config.win_len: [scalar] (default: 0.025)
%           window length in seconds used for calculating the spectrogram
%       config.win_shift: [scalar] (default: 0.01)
%           window shift in seconds when calculating the spectrogram
%       config.num_speech_elems: [scalar] (default: 30)
%           number of dictionary elements matrix factorization will use for
%           speech
%       config.num_noise_elems: [scalar] (default: 50)
%           number of dictionary elements matrix factorization will use for
%           noise
%       config.speech_sparsity: [scalar] (default: 0)
%           sparsity level of the encoding matrix for speech
%       config.noise_sparsity: [scalar] (default: 0)
%           sparsity level of the encoding matrix for noise
%       config.frequency_regularization: [matrix] (default: (nfft/2+1)-by-4 matrix filled with 1e7)
%           matrix of (nfft/2+1) rows indicating how similar each frequency
%           bin of the noise dictionary from running matrix factorization
%           on the noisy signal should be to the noise dictionary from
%           running matrix factorization on the noise-only signal. Higher
%           values enforce greater similarity/less change. Multiple columns
%           allow for different frequency regularizations applied simultaneously.
%       config.temporal_regularization: [scalar] (default: 100)
%           how much to use the noise statistics learned after running
%           matrix factorization on the noise-only signal
%       config.maxiter: [scalar] (default: 200)
%           number of iterations to use for updating the matrix
%           factorization equations
%       -------------------------------------------
%       Automatic Speech Start Detection Parameters
%       -------------------------------------------
%       config.threshold: [scalar] (default: 0.03)
%           spectrogram bins with power less than threshold x total_power
%           will be used to detect speech start. Want to estimate speech
%           start from frequency bins with low noise power.
%       config.power_dev: [scalar] (default: 2.3)
%           minimum number of standard deviations greater than the mean
%           power to consider a frame as speech
%       config.num_power_est_frames: [scalar] (default: 20)
%           number of beginning frames of noisy spectrogram to calculate
%           freq. bin power stats
%
% Outputs:
%   denoised_sig: [vector]
%       denoised speech. Will be at the same sampling rate as noisy_sig (fs).
%   noise_hat: [vector]
%       noise estimated from the noisy signal. Will be at the same
%       sampling rate as noisy_sig (fs).
%   cost_noisy: [vector]
%       value of the matrix factorization cost function at each iteration
%

% Setup parameters
if nargin < 4
    config = struct;
end
if ~isfield(config, 'nfft')
    config.nfft = 1024;
end
if ~isfield(config, 'win_len')
    config.win_len = 0.025;
end
if ~isfield(config, 'win_shift')
    config.win_shift = 0.01;
end
if ~isfield(config, 'num_speech_elems')
    config.num_speech_elems = 30;
end
if ~isfield(config, 'num_noise_elems')
    config.num_noise_elems = 50;
end
if ~isfield(config, 'speech_sparsity')
    config.speech_sparsity = 0;
end
if ~isfield(config, 'noise_sparsity')
    config.noise_sparsity = 0;
end
if ~isfield(config, 'frequency_regularization')
    config.frequency_regularization = 1e7*ones(config.nfft/2 + 1, 4);
elseif size(config.frequency_regularization, 1) ~= config.nfft/2 + 1
    error(['Frequency regularization has incorrect dimensions']);
end
if ~isfield(config, 'temporal_regularization')
    config.temporal_regularization = 100;
end
if ~isfield(config, 'maxiter')
    config.maxiter = 200;
end
if ~isfield(config, 'threshold')
    config.threshold = 0.03;
end
if ~isfield(config, 'power_dev')
    config.power_dev = 2.3;
end
if ~isfield(config, 'num_power_est_frames')
    config.num_power_est_frames = 20;
end

% Spectrogram params
win_samps = round(config.win_len * fs);
shift_samps = round(config.win_shift * fs);

% Randomly initialize speech dictionary
w_speech = rand(config.nfft/2 + 1, config.num_speech_elems);
w_speech = w_speech * diag(1 ./ sqrt(sum(w_speech.^2, 1)));

% Preprocess the noisy signal
noisy_sig = noisy_sig(:);
noisy_sig = noisy_sig - mean(noisy_sig);
noisy_sig = noisy_sig / max(abs(noisy_sig));

% Create spectrogram for noisy signal
noisy_spec = spectrogram(noisy_sig, win_samps, win_samps-shift_samps, config.nfft, fs, 'yaxis');
[m, n] = size(noisy_spec);

% Calculate the noise signal estimate
if isscalar(noise_est)
    if noise_est <= 0
        % Need to automatically detect
        % start of speech, noise spectrogram will be the first num_frames frames of
        % the noisy spectrogram before speech starts.
        power_per_bin = sum(abs(noisy_spec(:, 1:config.num_power_est_frames).^2), 2) / config.num_power_est_frames;
        total_power = sum(power_per_bin);
        low_power_bins = find(power_per_bin < config.threshold*total_power);
        if isempty(low_power_bins)
            error('Threshold too low!');
        end    
        speech_start = zeros(length(low_power_bins), 1);
        for bin = 1 : length(low_power_bins)
            inp = abs(noisy_spec(low_power_bins(bin), :));
            found_idx = find(inp(config.num_power_est_frames+1:end) > config.power_dev*std(inp(1:config.num_power_est_frames)) + mean(inp(1:config.num_power_est_frames)), 1) + config.num_power_est_frames;
            if isempty(found_idx)
                speech_start(bin) = n;
            else
                speech_start(bin) = found_idx;
            end
        end
        speech_start_idx = round(median(speech_start));
        sig_len = floor(((speech_start_idx - 2) * config.win_shift + config.win_len) * fs);
        noise_sig = noisy_sig(1 : sig_len);
        display(['Automatically detected speech at ', num2str(sig_len / fs), ' s']);
    else
        if noise_est > length(noisy_sig)
            warning('The duration specified to estimate the noise is greater than the duration of noisy signal. Using duration equal to the duration of the noisy signal.');
            noise_est = length(noisy_sig);
        elseif noise_est < 0.5
            warning('The duration of the noise estimate is short (less than 0.5 seconds). This can degrade the denoising performance. Consider using a longer noise estimate for better performance.');
        end
        noise_sig = noisy_sig(1 : round(noise_est*fs));
    end
elseif isvector(noise_est)
    if length(noise_est) < 0.5*fs
        warning('The duration of the noise estimate is short (less than 0.5 seconds). This can degrade the denoising performance. Consider using a longer noise estimate for better performance.');
    end
    noise_sig = noise_est(:);
    
    % Determine if noise estimate is shorter or longer than the noisy
    % signal
    min_len = min(length(noisy_sig), length(noise_sig));
    
    % Scale and shift noise estimate to minimize MSE between noisy signal
    % and noise estimate
    noisy_sig_trunc = noisy_sig(1 : min_len);
    noise_sig_trunc = noise_sig(1 : min_len);
    scaling = (noisy_sig_trunc' * noise_sig_trunc - (noisy_sig_trunc' * ones(min_len, 1)) * (noise_sig_trunc' * ones(min_len, 1)) / (ones(1, min_len) * ones(min_len, 1))) / (noise_sig_trunc' * noise_sig_trunc - (noise_sig_trunc' * ones(min_len, 1)) * (noise_sig_trunc' * ones(min_len, 1)) / (ones(1, min_len) * ones(min_len, 1)));
    shift = (noisy_sig_trunc' * ones(min_len, 1) - scaling * noise_sig_trunc' * ones(min_len, 1)) / (ones(1, min_len) * ones(min_len, 1));
    noise_sig = scaling * noise_sig + shift;
else
    error('Input noise estimate is not a vector or scalar.');
end

% Create spectrogram for noise estimate.
noise_spec = spectrogram(noise_sig, win_samps, win_samps-shift_samps, config.nfft, fs, 'yaxis');
num_noise_frames = size(noise_spec, 2);

% Run NMF on noise estimate to get a noise dictionary and encoding matrix
noise_est_opts = struct;
noise_est_opts.H_init = exp(randn(config.num_noise_elems, num_noise_frames));
noise_est_opts.maxiter = config.maxiter;
[w_noise, h_noise] = nmf(abs(noise_spec), config.num_noise_elems, noise_est_opts);

% Calculate noise statistics
target_mean = mean(log(h_noise), 2);
target_var = var(log(h_noise), [], 2);

% Run CMF-WISA with additional regularization terms on the noisy signal
noisy_opts = struct;
noisy_opts.W_init = {w_speech w_noise};
noisy_opts.H_init = {rand(config.num_speech_elems, n); [h_noise exp(diag(sqrt(target_var))*randn(config.num_noise_elems, n-num_noise_frames) + repmat(target_mean, 1, n-num_noise_frames))]};
noisy_opts.H_sparsity = {config.speech_sparsity; 0};
noisy_opts.W_regularization = {zeros(config.nfft/2 + 1, 1) config.frequency_regularization};
noisy_opts.H_regularization = {zeros(1, 1); config.temporal_regularization};
noisy_opts.maxiter = config.maxiter;
[w_noisy, h_noisy, p_noisy, cost_noisy] = cmfwisa_regularized(noisy_spec, {config.num_speech_elems config.num_noise_elems}, {rand(config.nfft/2 + 1, config.num_speech_elems) w_noise}, {rand(config.num_speech_elems, size(noisy_spec, 2)); h_noise}, noisy_opts);

% Reconstruct the estimated speech and noise
speech_spec_sirmax = ReconstructFromDecomposition(w_noisy{1}, h_noisy{1}) .* p_noisy{1};
noise_spec_sirmax = ReconstructFromDecomposition(w_noisy{2}, h_noisy{2}) .* p_noisy{2};
noise_softmask = abs(noise_spec_sirmax) ./ (abs(speech_spec_sirmax) + abs(noise_spec_sirmax));
noise_spec_sarmax = noise_softmask .* noisy_spec;
noise_hat = ReconstructSignal(abs(noise_spec_sarmax), angle(noise_spec_sarmax), config.nfft, ...
                                config.win_len, config.win_shift, fs, length(noisy_sig));
speech_hat = noisy_sig - noise_hat;

end  % function
