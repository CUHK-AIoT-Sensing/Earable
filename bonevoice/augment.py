from utils.bcf import Bone_Conduction_Function

if __name__ == "__main__":
    dataset_name = 'EMSB'  # or 'EMSB', 'V2S'
    bcf = Bone_Conduction_Function(dataset_name)

    # dataset = ABCS_dataset()
    # bcf = Bone_Conduction_Function(dataset, 'ABCS')

    bcf.plot_bcf()
    # bcf.extraction()
    # bcf.plot_reconstruction(10)

