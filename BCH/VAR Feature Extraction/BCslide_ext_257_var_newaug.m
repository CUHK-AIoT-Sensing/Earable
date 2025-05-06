function word_feac = BCslide_ext_257_var_newaug(vocabulary,ifsen,frame, info_band, ar_lags,var1,speechlen,stride,ifaug,ifvar,txside)
    % This function extracts features from the original signal and its augmented versions
    % Input:
    %   vocabulary: a cell array of size (num_words, num_augmented+3)
    %       vocabulary{i,1} = numeric label
    %       vocabulary{i,2} = string label
    %       vocabulary{i,3} = original signal
    %   fmcwchirp: 1 if the detection signal is FMCW chirp, 0 if it is OFDM
    %   ar_lags: number of AR coefficients to extract
    %   var1: order of difference
    %   speechfil: pass band (khz) of speech filter
    %   speechlen: length of speech in frames
    %   stride: stride of sliding window in frames

    fsr = 48000;
    [li,ll]=size(vocabulary);

    % ifaug by default is 1, if set to 0, only original signal is used
    if nargin<8
        ifaug=0;
    end

%     if ifsen
%         frame = 2400;
% %         info_band = 815:1110;%815:1110 17.5kHz = 877
%     %     tx=txo(1:1+frame);
%         txf = ones([frame/2+1,1]);
%     else
        % frame = 512; %2046; 5080; 254
        % info_band = 192:247; %748:980;%748:980;% 
        % info_band = round(99*frame/254):round(120*frame/254);
%         tx=txo(455:454+frame);

        % [txo,fst]=audioread('ofdm_17823khz_257_paprconfirm.wav');%ofdm_qpsk_demo_onec_50ms
        % tx=txo(713:713+511);%3420:5465 1225:1225+512-1
        % txf = rfft(tx);%% ones([frame/2+1,1])
    % end
    
    
    alp_fea=[];
    word_feac={};
    chan_resp_silent=[];
    labelY=[];


    for j=3:max(ll,3)
        tic
        parfor i=1:li
            rx_original = vocabulary{i,j};

            if ifsen && isa(vocabulary{i,1},'char') %%% skip 0000 and 9999
                features_row = [vocabulary{i,1}, repmat({[]},1,22)];
                word_feac = [word_feac; features_row];
                continue
            end
            if isempty(rx_original)
                continue
            end

            if size(rx_original, 2) ~= 2
                error('Input signal must have 2 channels');
            end
            rx_original = highpass(rx_original,17000,48000,Steepness = 0.9999);
            % % Split into left and right tx signals by filter
            % rx_lefttx = highpass(rx_original,20050,48000,Steepness = 0.9999,StopbandAttenuation = 100);
            % rx_righttx = bandpass(rx_original,[17000 20050],48000,Steepness = 0.9999);
            % rx_original = [rx_lefttx, rx_righttx]; % lefttx_downmic, lefttx_upmic, righttx_downmic, righttx_upmic


            % Validate that the signal has 2 channels

            % chan_wid = floor(length(rx_original)/frame);
            % Extract features from original signal
            arma_coef_original = slwinAR(rx_original, frame, ar_lags, var1, speechlen, stride, ifvar);
            % resize the original signal to matrix of size (frame, chan_wid)
            % try
            % rx_resized = reshape(rx_original(1:end-mod(length(rx_original),frame),:),frame,[],size(rx_original,2));
            % catch
            %     disp("error")
            % end
            
            % Augmentation 1: Resampling
            arma_coef_resampled = {};
            if ifaug
                arma_coef_offset = {};
                arma_coef_offset(1) = arma_coef_original;
                % for k=1:stride-1
                %     offset = k; %mod((stride-mod(k-1,stride)),stride); % different offset for sliding window augmentation
                %     rx_offset = rx_original((offset+1)*frame+1:end,:);
                %     % rx_offset = rx_resized(:,offset+1:end,:);
                %     arma_coef_offset = [arma_coef_offset, slwinAR(rx_offset, frame, ar_lags, var1, speechlen, stride, ifvar)];
                %     arma_coef_resampled = [arma_coef_resampled, arma_coef_offset{end}];
                % end
                
                for k=1:6
                    offset = mod((stride-mod(k-1,stride)),stride); % different offset 
                    % rx = rx_resized(:,offset+1:end,:);
                    ar_coef = arma_coef_original{1}; % arma_coef_offset{offset+1};
                    rate = 0.8 + rand()*0.4; % random between 0.8 and 1.2
                    
                    % rx_resampled = resample(rx, round((chan_wid-offset)*rate), chan_wid-offset,dim = 2);
                    % arma_coef_resampled = [arma_coef_resampled, slwinAR(rx_resampled, frame, ar_lags, var1, speechlen, stride, ifvar)];
                    ar_coef_resam = resample(ar_coef, round(rate*size(ar_coef,1)), size(ar_coef,1), dim = 1);
                    arma_coef_resampled = [arma_coef_resampled, {ar_coef_resam}];
                end
    
                % Augmentation 2: Removing 10% consecutive timesteps
                for k=1:6
                    offset = mod((stride-mod(k-1,stride)),stride);
                    % rx = rx_resized(:,offset+1:end,:);
                    % start_idx = randi([1, floor(0.9*size(rx,2))]);
                    % end_idx = start_idx + floor(0.1*size(rx,2));
                    % rx_removed = [rx(:,1:start_idx,:) rx(:,end_idx+1:end,:)];
                    % arma_coef_resampled = [arma_coef_resampled, slwinAR(rx_removed, frame, ar_lags, var1, speechlen, stride, ifvar)];
                    ar_coef = arma_coef_original{1}; %arma_coef_offset{offset+1};
                    start_idx = randi([1, floor(0.9*size(ar_coef,1))]);
                    end_idx = start_idx + floor(0.1*size(ar_coef,1));
                    ar_coef_removed = [ar_coef(1:start_idx,:,:); ar_coef(end_idx+1:end,:,:)];
                    arma_coef_resampled = [arma_coef_resampled, {ar_coef_removed}];
                end
    
                % % Augmentation 3: Removing 10% random timesteps
                % for k=1:5
                %     offset = mod((stride-mod(k-1,stride)),stride);
                %     rx = rx_resized(:,offset+1:end,:);
                %     mask = ones(size(rx,2), 1);
                %     mask(randperm(size(rx,2), floor(0.1*size(rx,2)))) = 0;
                %     rx_random_removed = rx(:,logical(mask),:);
                %     arma_coef_resampled = [arma_coef_resampled, slwinAR(rx_random_removed, frame, ar_lags, var1, speechlen, stride, ifvar)];
                % end
    
                % % Augmentation 4: Adding Gaussian noise
                % for k=1:5
                %     offset = mod((stride-mod(k-1,stride)),stride);
                %     rx = rx_resized(:,offset+1:end,:);
                %     % calculate the standard deviation of every row of rx
                %     std_noise = 0.05 * median(std(rx, 0, 2));
                %     % generate Gaussian noise with the same size as rx, apply the standard deviation for each row
                %     noise = std_noise .* randn(size(rx));
                %     rx_noisy = rx + noise;
                %     arma_coef_resampled = [arma_coef_resampled, slwinAR(rx_noisy, frame, ar_lags, var1, speechlen, stride, ifvar)];
                % end

            end
            % Collect all features in a row
            features_row = [vocabulary{i,1}, vocabulary{i,2}, arma_coef_original, arma_coef_resampled]; % ... add other augmented features

            % Add the feature row to the dataset matrix
            word_feac = [word_feac; features_row];
        end
        used_time=toc;
        disp(strcat(string(j),'/',string(max(ll,3)),' takes ',string(used_time),' seconds'));
    end
end

function fea = slwinAR(rx_ori, frame_full, ar_lags, var1, speechlen, stride, ifvar)
%%% auxiliary function of sld_aug 
        %%%%%% if it is the augment 
        frame = round(frame_full);
        if mod(frame_full,1)==0
            disp("wrong frmae")
        end

        if length(size(rx_ori))>1 && length(size(rx_ori))>2%%%size(rx_ori,2) ~= 1

            if ~ismember(frame, size(rx_ori)) 
                disp("error: frame size not match in slwinAR")
                disp(size(rx_ori))
                fea = {[]};
                return
            end
            if size(rx_ori,1)~=frame && size(rx_ori,2)==frame
                rx_ori = rx_ori';
            end
            % rx = rx_ori(:);
            rx = reshape(rx_ori,[],size(rx_ori,3));
        else
            rx = rx_ori;
        end

        chan_wid = floor(length(rx)/max(frame_full,frame));
        Ttmp = floor((chan_wid-speechlen)/stride)+1; % [123]45, 12[345] -> (5-3)/2+1 =2 
        ahead = 6;
% %%%%%%  detect speech            
        % [idxspeech,rxfil] = detectvoice(rx,speechfil,fsr);
        % try
        %     idxss_6 = ceil((idxspeech(1)-frame*ahead)/frame);
        %     if idxss_6 <= 0
        %        Ttmp = Ttmp + idxss_6 - 1;
        %     end
        %     idxss = max(idxss_6,1);
        % 
        % catch
        %     disp(idxspeech)
        % end
        % speechadd2 = ceil(((idxspeech(2)-1)/frame))-idxss+ahead;
        % if idxss + speechlen - 1 > chan_wid %max(speechlen,speechadd2)
        %     % disp(strcat('error idxss too late, i =',i,', j =',j))
        %     continue
        % end

        idxss = 1;
        % if idxss + speechlen + Ttmp -2 > chan_wid
        %     Ttmp = chan_wid - (idxss + speechlen -1) + 1;
        % end
        if Ttmp <= 0
            disp("Ttmp <= 0")
            fea = {[]};
            return 
        end
 %%%%%%%%%%%%%% Channel response
        % chan_resp=[];
        % parfor i1=0:floor(length(rx)/frame)-1
        %     rxf = rfft(rx(i1*frame+1:i1*frame+frame).*hann(frame));%
        %     chan_resp=[chan_resp,rxf(info_band)./txf(info_band)];
        % end
        
        % rx = highpass(rx,15000,48000);
        % rxset = zeros(Ttmp,speechlen*frame-var1,size(rx,2));
        rxset = zeros(Ttmp,speechlen*frame,size(rx,2));
        % rxset = {Ttmp,1};
        
        d_rx_all = diff([rx; rx(1:var1,:)],var1);
        for ii = 1:Ttmp
            idx_start = round((idxss+ii-2)*stride*frame_full)+1;
            idx_end = idx_start-1 + speechlen*frame;
            d_rx = d_rx_all(idx_start:idx_end,:);

            % rxtmp = rx(idx_start+1:idx_end,:);
            %%% difference taken in function because we do augmentation first 
            % d_rx = diff(rxtmp,var1);
            try
                if size(d_rx)>1
                    rxset(ii,:,:) = d_rx;
                    % rxset{ii} = d_rx;
                else
                    rxset(ii,:) = d_rx;
                end
            catch
                disp("cat error")
            end
        end

%%%%%%%%%%%%%%%%%%%%%%%%% AutoRegressive
        % % EstMdl = estimate(arima(4,0,0),rx((idxss-1)*frame+1:(idxss+speechlen)*frame),'Display',"off");
        % % arma_coef = [cell2mat(EstMdl.AR),cell2mat(EstMdl.MA)];
        if ~ifvar
            % arma_coef = arburg(rxset',ar_lags); % for 1-channel
            arma_down = arburg(rxset(:,:,1)',ar_lags);
            arma_up = arburg(rxset(:,:,2)',ar_lags);
            arma_coef(:,:,1)=arma_down;
            arma_coef(:,:,2)=arma_up;
            arma_coef(:,1,:) = [];
        else
            %%% VAR Coefficients
            Mdl = varm(2, ar_lags);
            arma_coef = NaN(Ttmp,ar_lags,4);
            % arma_coef = NaN(Ttmp,ar_lags,8);
            for ii = 1:Ttmp
                %%%% for right Rx using MATLAB method
                RTx_EstMdl = estimate(Mdl,squeeze(rxset(ii,:,1:2)), 'Display',"off"); % rxset{ii}
                arma_coef(ii,:,:) = reshape(cell2mat(RTx_EstMdl.AR),4,ar_lags)';
                % arma_coef(ii,:,1:4) = zeros(100,4);

                % [~,~,R]=lic(squeeze(rxset(ii,:,1:2))',ar_lags);
                % arma_coef(ii,:,1:4) = reshape(R,4,ar_lags)';

                % arma_coef = 0;
    
                %  LTx_EstMdl = estimate(Mdl,squeeze(rxset(ii,:,1:2)), 'Display',"off");
                % arma_coef(ii,:,5:8) = reshape(cell2mat(LTx_EstMdl.AR),4,ar_lags)';
                
            end
        end
        fea = {arma_coef};%s_ac_a;%[cird_ls(:);s_ac_a(:)];

    % return the AR coefficients
end


function rfft = rfft(a)
     ffta = fft(a);
     rfft = ffta(1:(floor(length(ffta)/2)+1));
end
    %   speechlen: length of speech in frames