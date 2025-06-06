from utils.vib_dataset import EMSB_dataset, ABCS_dataset, V2S_dataset
from utils.bcf import Bone_Conduction_Function

if __name__ == "__main__":
    dataset = EMSB_dataset()
    bcf = Bone_Conduction_Function(dataset)
    # bcf.extraction()
    # bcf.prediction()
    bcf.aggregation()