%% Metadata for Train set
clear
clc

addpath data
% load('IReqHOA_[7 9 3].mat')

feat = 'IV';            % feature type
Npos = 16;              % # spk-mic position
Nspeech = 1200;         % # speeches for each position
Ndata = 4620;           % # total TIMIT train set
Nfeat = 150; 			% # extracted feature

path_data = dir('data/TIMIT/TRAIN/**/*.wav');

len_file = zeros(Ndata,1);
for ii = 1:Ndata
    len_file(ii,1) = length(path_data(ii).name);
end
len_max = 27 + max(len_file);

path_all_speech = [];
path_ = cell(Ndata,1);
for ii = 1:Ndata
    name_folder = path_data(ii).folder;
    name_folder = split(name_folder, '\');
    name_file   = path_data(ii).name;
    
    path_ = cell2mat(join([name_folder(3:end)' name_file], '\'));
    blank = blanks(len_max - length(path_));
    path_all_speech = [path_all_speech; path_ blank];
end

rooms = 'room1';
n_loc = Npos;

% path_feat = dir(['data/' feat '_fitting/TRAIN_' num2str(Npos) '_' num2str(Nspeech) '/*.npz']);
path_feat = dir(['data/' feat '_fitting/VALID_' num2str(Npos) '/*.npz']);
list_fname = [];
for ii = 1:length(path_feat)
    list_fname = [list_fname; path_feat(ii).name];
end

% save(['data/' feat '_fitting/TRAIN_' num2str(Npos) '_'  num2str(Nspeech) '/metadata.mat'],...
%     'n_loc', 'rooms', 'path_all_speech', 'list_fname', '-v6');
save(['data/' feat '_fitting/VALID_' num2str(Npos) '/metadata.mat'],...
    'n_loc', 'rooms', 'path_all_speech', 'list_fname', '-v6');

%% Metadata for Test set
clear
clc

addpath data

feat = 'IV';
Ndata = 1680;
Nfeat = 805;
path_data = dir('data/TIMIT/TEST/**/*.wav');

len_file = zeros(Ndata,1);
for ii = 1:Ndata
    len_file(ii,1) = length(path_data(ii).name);
end
len_max = 26 + max(len_file);

path_all_speech_t = [];
path_ = cell(Ndata,1);
for ii = 1:Ndata
    name_folder = path_data(ii).folder;
    name_folder = split(name_folder, '\');
    name_file   = path_data(ii).name;
    
    path_ = cell2mat(join([name_folder(3:end)' name_file], '/'));
    blank = blanks(len_max - length(path_));
    path_all_speech_t = [path_all_speech_t; path_ blank];
end

rooms = 'room1';
n_loc = 8;

path_feat = dir(['data/' feat '_fitting/TEST_8/*.npz']);
list_fname = [];
for ii = 1:Nfeat
    list_fname = [list_fname; path_feat(ii).name];
end

save(['data/' feat '_fitting/TEST_8/metadata.mat'],...
    'n_loc', 'rooms', 'path_all_speech', 'list_fname', '-v6');

save('metadata.mat', 'fs', 'l_frame','l_hop','list_fname','n_fft',...
    'n_freq','n_loc','path_all_speech','rooms','-v6');
