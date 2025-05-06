function tone_start_index = synctone(rx)

    Fs = 48000;  % sample rate in Hz, adjust as needed
    tone_freq = 2000;  % tone frequency in Hz
    tone_duration = 0.5;  % duration of tone in seconds, adjust as needed
    detect_duration = 25;

    % Create a time vector for one period of the 2kHz tone
    t = 0:1/Fs:tone_duration-1/Fs;
    
    % Create the template signal for the 2kHz tone
    tone = sin(2*pi*tone_freq*t);
    
    rxlow = bandpass(rx(1:Fs*detect_duration),[1900,2100],Fs);
    rxlow = abs(rxlow);
    tone = abs(tone);
    % Cross-correlate the received signal with the tone template
    [correlation,lags] = xcorr(rxlow, tone);
    
    % The start index of the tone corresponds to the maximum of the correlation
    [~, tone_start_index] = max(abs(correlation));
    
    % Adjust for the length of the tone template
    % tone_start_index = tone_start_index - length(tone_template) + 1;
    tone_start_index = lags(tone_start_index);
end