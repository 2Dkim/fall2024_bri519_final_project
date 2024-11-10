%This program conducts an analysis of power spectrum of LFP data 
%recorded from mouse auditory cortex in response to tones.
%It assumes that the data file is in the same directory as the program
%It produces figures
%Who wrote this and how do the reach them?
%When?
%Version history

%% 0 Init - start from a blank slate, then define constants
clear all
close all
clc

scrSz = get(0,'ScreenSize'); %Get screensize
cutoffFrequency = 1e3; %Low pass cutoff frequency
binWidth = 5; %5 Hz bands
maxFreq = 200; %Plot signal power up to 200 Hz
numTrials = 200; %We have data from 200 trials
numSessions = 4; %We have data from this many sessions
fs = 1e4; %Sampling frequency
stimOnset = 1000; %Stimulus comes on at 100 ms.
stimOffset = 1500; %Stimulus goes off at 150 ms.

%Power spectrum windowing parameters:
wind = hanning(256); %A 256 element hanning window will do
overl = 255; %Maximal overlap
nfft = [0:binWidth:maxFreq]; %Create the frequency bins

%% 1 Loader

load('mouseLFP.mat') %Loading data file
dataSamples = size(DATA{1,1},2); %We have data from this time points per trial

%% 2 Filtering 

% The original data was sampled at a frequency of 10 kHz.
% To get the LFP, we run a low-pass butterworth filter with a cutoff frequency of 1 kHz.
nyquistFrequency = fs/2; %Nyquist frequency
lowPassData = zeros(numSessions, numTrials,dataSamples); % Preallocate matrix of zeros for low-pass data
[B,A] = butter(5,(cutoffFrequency/nyquistFrequency),'low'); % Create a 5th order low-pass butterworth filter

for ii = 1:numSessions %Go through all sessions
    for jj = 1:numTrials%Go through all trials
       lowPassData(ii,jj,:) = filtfilt(B,A,DATA{ii,1}(jj,:)); % Low pass filter session data
    end
end



%% 3 Formatting

%Find indices of high and low tones
for ii = 1:numSessions
    ToneIndices{ii,1} = find(DATA{ii,5} == min(unique(DATA{ii,5}))); % indices of low tones    
    ToneIndices{ii,2} = find(DATA{ii,5} == max(unique(DATA{ii,5}))); % indices of high tones

    filteredTones{ii,1} = squeeze(lowPassData(ii,ToneIndices{ii,1},:)); % Find filtered data for low tones
    filteredTones{ii,2} = squeeze(lowPassData(ii,ToneIndices{ii,2},:)); % Find filtered data for high tones
end


%% 4 Calculating means and spectrograms for all sessions

%a) Means
for ii = 1:numSessions
     meanLowTones(ii,:) = mean(filteredTones{ii,1}); % average low tone response
     meanHighTones(ii,:) = mean(filteredTones{ii,2}); % average high tone response
end

minimin = min(min([meanLowTones meanHighTones])); %Overall lowest value
maximax = max(max([meanLowTones meanHighTones])); %Overall highest value

%b) Spectrograms
%S = zeros(numSessions,2,length(nfft),dataSamples-overl); %Preallocate matrix to contain spectrograms
%for ii = 1:numSessions
 %   for jj = 1:2
 %       if jj == 1
 %   [S(ii,jj,:,:) F T] = spectrogram(meanLowTones(ii,:),wind,overl,nfft,fs,'yaxis'); 
 %   else
 %   [S(ii,jj,:,:) F T] = spectrogram(meanHighTones(ii,:),wind,overl,nfft,fs,'yaxis'); 
 %   end
 %   end
%end
%% 5 Plotting everything
for ii = 1:numSessions
    figure %Open a new figure
    set(gcf,'Position',scrSz); %Make full screen
    for jj = 1:2
      %Plot mean LFPs
      subplot(2,2,jj); % low tone ERP plot
      h1 = errorband_2015(filteredTones{ii,jj}); % use the errorband function to plot +/- 1 SEM
      set(h1,'linestyle','none');
      hold on;
      if jj == 1
        plot(meanLowTones(ii,:),'color','k'); % plot the mean LFP for low tone
      else
        plot(meanHighTones(ii,:),'color','k'); % plot the mean LFP for high tone        
      end
    
      %Put x axis in right time base, taking sampling rate into account
      temp = get(gca,'XTick');
      temp2 = temp/fs*1000;
      set(gca,'XTickLabel',temp2)
      
      title(['Mean LFP from Mouse Auditory Cortex (Session ',num2str(ii),') in response to tone ',num2str(jj)]);
      xlabel('Time (ms)');
      ylabel('Membrane Potential (mV)');
      ylim([minimin*1.1 maximax*1.1]); %Put all on the same scale
      
      %Plot spectrograms
      subplot(2,2,jj+2)    
      if jj == 1
        spectrogram(meanLowTones(ii,:),wind,overl,nfft,fs,'yaxis'); 
      else
        spectrogram(meanHighTones(ii,:),wind,overl,nfft,fs,'yaxis'); 
      end

      title(['Spectrogram of LFP data in response to tone ', num2str(jj)]);
      colormap(jet)
      h = colorbar;
      set(h,'Limits',[-88.8 6.2])
      caxis([-88.8 6.2])
    end
    set(gcf,'color','w')
end
    

%To do: 
%*Clarify whether it is better to do the spectrogram on the average trace or
%on all traces, then take the average spectrogram
%*Find the overall mean for the colorbar. Instead of hardcoding it
%*Separate calculating and plotting 
