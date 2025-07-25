import numpy as np
import torch
import core.utils as utils
import torch.utils.data as data
import os
import json
import torch.nn as nn

class NpyFeature(data.Dataset):
    def __init__(self, data_path, audio_path, mode, modal, feature_fps, n_segments, sampling, class_dict, seed=-1, supervision='weak', default_fps=None):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.n_segments = n_segments
        self.default_fps = default_fps

        if self.modal == 'all':
            self.feature_path = []
            # modified by xuwb 20241206 start
            '''
            # load audio and video data
            for _modal in ['rgb', 'flow']:
                self.feature_path.append(os.path.join(data_path, 'feats/tsn/', _modal, self.mode))
            '''
            self.feature_path.append(os.path.join(data_path, 'feats/tsn/rgb/', self.mode))
            self.feature_path.append(os.path.join(audio_path, 'feats/audio_context/', self.mode))
            # modified by xuwb 20241206 end
        else:
            self.feature_path = os.path.join(data_path, 'features', self.mode, self.modal)
        self.class_name_to_idx = class_dict
        self.n_classes = len(self.class_name_to_idx.keys())

        self.supervision = supervision
        self.sampling = sampling
        anno_path = os.path.join(data_path, 'annotations/metadata.json')
        dict_db = self._load_json_db(anno_path)
        self.anno_dict = dict_db

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            if(line.strip() not in dict_db):
                continue
            self.vid_list.append(line.strip())
        split_file.close()
        print('=> {} set has {} videos'.format(mode, len(self.vid_list)))

        # anno_file = open(anno_path, 'r')
        # self.anno = json.load(anno_file)
        # anno_file.close()
        
        # bkg_path = os.path.join(data_path, 'backgrounds.json')
        # bkg_file = open(bkg_path, 'r')
        # self.bkg = json.load(bkg_file)
        # bkg_file.close()


    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        # modified by xuwb 20241207 start
        '''
        # return rgb_feature and audio_feature, for multimodal fusion
        '''
        # data, vid_n_seg, sample_idx = self.get_data(index)
        video_data, audio_data, vid_n_seg, sample_idx = self.get_data(index)
        # modified by xuwb 20241207 end
        label, fps, temp_anno = self.get_label(index, vid_n_seg, sample_idx)
        # if self.mode=='train':
        #     bkg_anno = self.get_bkg(index, vid_n_seg, sample_idx)
        # else:
        #     bkg_anno = torch.Tensor(0)

        # modified by xuwb 20241207 start
        '''
        # return rgb_feature and audio_feature, for multimodal fusion
        '''
        # return data, label, temp_anno, self.vid_list[index], vid_n_seg
        return video_data, audio_data, label, temp_anno, self.vid_list[index], vid_n_seg, fps
        # modified by xuwb 20241207 end
    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_db = json.load(fid)

        # fill in the db (immutable afterwards)
        dict_db = {'id': {
                         'fps': None,
                         'duration': None,
                         'split': None,
                         'segments': None,
                         'labels': None,
                         'valid_frames': None
                         }
                    }
        for value in json_db:
            key = os.path.splitext(os.path.basename(value['file']))[0]   # file name without ext
            # skip the video if not in the split
            if value['split'].lower() not in self.mode:
                continue
            
            if isinstance(self.feature_path, list):
                assert len(self.feature_path) == 2
                feat_file = os.path.join(self.feature_path[0], key + '.npy')
            else:
                feat_file = os.path.join(self.feature_path, key + '.npy')
            if not os.path.exists(feat_file):
                continue

            # added by xuwb 20241111 start
            # choose fake sample, and choose modify_video sample
            if self.mode == 'dev' and len(value['fake_periods']) == 0:
                continue
            if self.mode == 'dev' and (value['modify_audio'] == False or value['modify_video'] == False):
            # if self.mode == 'dev' and value['modify_video'] == False:
                continue
            if self.mode == 'train' and len(value['fake_periods']) != 0 and (value['modify_audio'] == False or value['modify_video'] == False):
            # if self.mode == 'train' and len(value['fake_periods']) != 0 and value['modify_video'] == False:
                continue
            # added by xuwb 20241111 end


            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            elif 'video_frames' in value:
                fps = value['video_frames'] / value['duration']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']


            valid_acts = value['fake_periods']
            num_acts = len(valid_acts)
            segments = np.zeros([num_acts, 2], dtype=np.float32)
            labels = np.zeros([self.n_classes], dtype=np.float32)
            for idx, act in enumerate(valid_acts):
                segments[idx][0] = act[0]
                segments[idx][1] = act[1]
            if num_acts > 0:
                labels[self.class_name_to_idx['fake']] = 1
            else:
                labels[self.class_name_to_idx['real']] = 1

            # get valid_frames
            if 'video_frames' in value:
                valid_frames = value['video_frames']
            elif 'feature_frames' in value:
                valid_frames = value['feature_frames']
            else:
                assert False, "Unknown valid frames."
            

            dict_db[key] = {
                         'fps': fps,
                         'duration': duration,
                         'split': value['split'].lower(),
                         'segments': segments,
                         'labels': labels,
                         'valid_frames': valid_frames
                         }

        return dict_db
    ## added by xuwb 20241104 start
    # only use rgb as input, remove the flow
    # =================================rgb start=======================================
    '''
    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_n_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npy')).astype(np.float32)
            # flow_feature = np.load(os.path.join(self.feature_path[1],
            #                         vid_name + '.npy')).astype(np.float32)

            vid_n_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            # flow_feature = flow_feature[sample_idx]

            # feature = np.concatenate((rgb_feature, flow_feature), axis=1)
            feature = rgb_feature
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_n_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

        return torch.from_numpy(feature), vid_n_seg, sample_idx
    '''
    # =================================rgb end=========================================
    # ===============================rgb+flow start=====================================
    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_n_seg = 0

        if self.modal == 'all':
            rgb_feature = np.load(os.path.join(self.feature_path[0],
                                    vid_name + '.npy')).astype(np.float32)
            audio_feature = np.load(os.path.join(self.feature_path[1],
                                    vid_name + '.npy')).astype(np.float32)

            audio_tensor = torch.tensor(audio_feature, dtype=torch.float32).unsqueeze(0)
            audio_tensor = audio_tensor.permute(0, 2, 1)
            output_audio_tensor = nn.AdaptiveAvgPool1d(rgb_feature.shape[0])(audio_tensor)
            output_audio_tensor = output_audio_tensor.permute(0,2,1)
            audio_feature = output_audio_tensor.squeeze(0).detach().numpy()
            vid_n_seg = rgb_feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(rgb_feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(rgb_feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            audio_feature = audio_feature[sample_idx]
            # modified by xuwb 20241207 start
            '''
            # return rgb_feature and audio_feature, for multimodal fusion
            feature = np.concatenate((rgb_feature, audio_feature), axis=1)
            '''
            
            # modified by xuwb 20241207 end
        else:
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_n_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(feature.shape[0])
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(feature.shape[0])
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]
        # modified by xuwb 20241207 start
        '''
        # return rgb_feature and audio_feature, for multimodal fusion
        '''
        # return torch.from_numpy(feature),vid_n_seg, sample_idx
        return torch.from_numpy(rgb_feature), torch.from_numpy(audio_feature), vid_n_seg, sample_idx
        # modified by xuwb 20241207 end
    # ===============================rgb+flow end=====================================    
    ## added by xuwb 20241104 end


    def get_label(self, index, vid_n_seg, sample_idx):
        vid_name = self.vid_list[index]
        
        anno_dict = self.anno_dict[vid_name]
        # label = np.zeros([self.n_classes], dtype=np.float32)
        label = anno_dict['labels']
        fps = anno_dict['fps']
        # classwise_anno = [[]] * self.n_classes

        # for _anno in anno_dict['segments']:
        #     # label[self.class_name_to_idx[_anno['label']]] = 1
        #     classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)

        if self.supervision == 'weak':
            return label, fps, torch.Tensor(0)
        # else:
        #     temp_anno = np.zeros([vid_n_seg, self.n_classes])
        #     t_factor = self.feature_fps / 16

        #     for class_idx in range(self.n_classes):
        #         if label[class_idx] != 1:
        #             continue

        #         for _anno in classwise_anno[class_idx]:
        #             tmp_start_sec = float(_anno['segment'][0])
        #             tmp_end_sec = float(_anno['segment'][1])

        #             tmp_start = round(tmp_start_sec * t_factor)
        #             tmp_end = round(tmp_end_sec * t_factor)

        #             temp_anno[tmp_start:tmp_end+1, class_idx] = 1

        #     temp_anno = temp_anno[sample_idx, :]

        #     return label, torch.from_numpy(temp_anno)
        
    def get_bkg(self, index, vid_n_seg, sample_idx):
        # modified by xuwb 20241013 start 
        # vid_name = self.vid_list[index]
        # bkg_list = self.bkg['results'][vid_name]

        # if True:
        #     temp_bkg = np.zeros([vid_n_seg])
        #     t_factor = self.feature_fps / 16

        #     for _bkg in bkg_list:
        #         tmp_start_sec = float(_bkg['segment'][0])
        #         tmp_end_sec = float(_bkg['segment'][1])

        #         tmp_start = round(tmp_start_sec * t_factor)
        #         tmp_end = round(tmp_end_sec * t_factor)

        #         temp_bkg[tmp_start:tmp_end+1] = 1

        #     temp_bkg = temp_bkg[sample_idx]

            # return torch.from_numpy(temp_bkg)
        # modified by xuwb 20241013 end
            return np.zeros([vid_n_seg])

    def uniform_sampling(self, length):
        if length <= self.n_segments:
            return np.arange(length).astype(int)
        samples = np.arange(self.n_segments) * length / self.n_segments
        samples = np.floor(samples)
        return samples.astype(int)

    def random_perturb(self, length):
        if self.n_segments == length:
            return np.arange(self.n_segments).astype(int)
        samples = np.arange(self.n_segments) * length / self.n_segments
        for i in range(self.n_segments):
            if i < self.n_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)
