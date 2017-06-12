function [out_sig] = ReconstructSignal(mag_spec, phase_spec, nfft, win_len, win_shift, fs, sig_len)
% ReconstructSignal Reconstruct time-domain signal from spectrogram.
% Assumes that the signal is real-valued, so the spectrogram contains
% values only for the first-half of the unit circle (this is the default
% behavior for Matlab's "spectrogram" function for real-valued input).
%
% Inputs:
%   mag_spec: matrix containing the magnitude spectrogram of the signal.
%   phase_spec: matrix containing the phase of the signal.
%   nfft: number of sampling points used to calculate the DFT (nfft-point
%       DFT)
%   win_len: window length in seconds used to create the spectrogram.
%   win_shift: window shift in seconds used to create the spectrogram.
%   fs: sampling rate of the signal.
%   sig_len: length (in samples) of the signal to be reconstructed
%       (optional).
%
% Output:
%   out_sig: vector containing time-domain signal of the spectrogram.

num_windows = size(mag_spec, 2);
win_samps = round(win_len * fs);
shift_samps = round(win_shift * fs);

% Calculate signal length if user did not specify this.
if nargin == 6
    sig_len = ceil(((num_windows - 1) * win_shift + win_len) * fs);
end

% Check if the spectrogram needs to be padded with zeros. Happens when the
% number of sampling points used to calculate the DFT (nfft) is less than
% the number of samples in the window (win_len*fs).
if nfft < win_samps
    if mod(win_samps, 2) == 0
        pad_amount = win_samps / 2 - nfft/2;
    else
        pad_amount = (win_samps+1) / 2 - nfft/2;
    end
    
    % Extend the magnitude and phase spectra around the entire unit circle
    mag_spec_ext = [mag_spec(1:nfft/2, :); zeros(2*pad_amount, num_windows); mag_spec(nfft/2:-1:2, :)];
    phase_spec_ext = [phase_spec(1:nfft/2, :); zeros(2*pad_amount, num_windows); -phase_spec(nfft/2:-1:2, :)];
else
    pad_amount = 0;
    
    % Extend the magnitude and phase spectra around the entire unit circle
    mag_spec_ext = [mag_spec; mag_spec(nfft/2:-1:2, :)];
    phase_spec_ext = [phase_spec; -phase_spec(nfft/2:-1:2, :)];
end

% Convert from frequency domain to time domain
sig_from_fft = real(ifft(mag_spec_ext .* exp(1j.*phase_spec_ext), nfft + 2*pad_amount));

% Use overlap-add to reconstruct the signal
out_sig = zeros(sig_len, 1);
undo_window = zeros(sig_len+shift_samps, 1);
for n = 1 : shift_samps : sig_len-win_samps+1
    out_sig(n : n+win_samps-1) = out_sig(n : n+win_samps-1) + sig_from_fft(1:win_samps, ceil(n/shift_samps));
    undo_window(n : n+win_samps-1) = undo_window(n : n+win_samps-1) + hamming(win_samps);
end
undo_window(n+shift_samps : n+shift_samps+win_samps-1) = undo_window(n+shift_samps : n+shift_samps+win_samps-1) + hamming(win_samps);
undo_window = undo_window(1 : sig_len);
undo_window(undo_window == 0) = 1;
out_sig = out_sig ./ undo_window;
% out_sig = out_sig - mean(out_sig);
% out_sig = out_sig / max(abs(out_sig));

end  % function
