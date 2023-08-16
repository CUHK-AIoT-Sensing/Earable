from dataset import EMSBDataset, ABCSDataset
import torch

def vad_annotation(audio):
    '''
    "In-Ear-Voice: Towards Milli-Watt Audio Enhancement With Bone-Conduction Microphones for In-Ear Sensing Platforms, IoTDI'23"
    '''
    pass

dataset = EMSBDataset('json/EMSB.json', ratio=0.8)
train_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=2, batch_size=4, shuffle=True)
print(len(dataset))

dataset = ABCSDataset('json/ABCS_train.json')
print(len(dataset))
# for data in train_loader:
#     print(type(data))