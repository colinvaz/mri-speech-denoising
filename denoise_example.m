clear;
clc;

% Set denoising parameters
% For details, type "help MRIdenoising" at the Matlab prompt
config.nfft = 1024;
config.win_len = 0.025;
config.win_shift = 0.01;
config.speech_sparsity = 1;
config.frequency_regularization = [1e7*ones(30, 1)                     1e7*ones(30, 1)                     1e7*ones(30, 1)                     1e7*ones(30, 1);
                                   1e6*ones(280, 1)                    1e5*ones(280, 1)                    1e4*ones(280, 1)                    1e3*ones(280, 1);
                                   1e7*ones(config.nfft/2+1-30-280, 1) 1e7*ones(config.nfft/2+1-30-280, 1) 1e7*ones(config.nfft/2+1-30-280, 1) 1e7*ones(config.nfft/2+1-30-280, 1)];
                               
% Load noisy speech
[noisy_speech, fs] = audioread('example.wav');

% Denoise speech
denoised_speech = MRIdenoising(noisy_speech, 0, fs, config);

% Plot noisy and denoised spectrograms
figure, spectrogram(noisy_speech, round(config.win_len * fs), round((config.win_len - config.win_shift) * fs), config.nfft, fs, 'yaxis');
figure, spectrogram(denoised_speech, round(config.win_len * fs), round((config.win_len - config.win_shift) * fs), config.nfft, fs, 'yaxis');

% Listen to denoised speech
%soundsc(denoised_speech, fs);
