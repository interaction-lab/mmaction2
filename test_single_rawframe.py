import torch

from mmaction.apis import init_recognizer, inference_recognizer

config_file = 'configs/recognition/tsm/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = 'checkpoints/tsm_k400_pretrained_r50_1x1x8_25e_hmdb51_rgb_20210630-10c74ee5.pth'

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device)

# test rawframe directory of a single video and show the result:
video = 'data/hmdb51/rawframes/eat/310ToYuma_eat_u_nm_np1_fr_med_4'
results = inference_recognizer(model, video)

print("results:\n", results)

# show the results

labels = open('tools/data/hmdb51/label_map.txt').readlines()

labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])
