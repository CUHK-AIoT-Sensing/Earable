import matplotlib.pyplot as plt
import numpy as np
import os
import sofa
from tqdm import tqdm
from random import uniform, sample 
from scipy import signal

def plot_coordinates(HRTF, title):
    coords = HRTF.Source.Position.get_values(system="cartesian")
    x0 = coords
    n0 = coords
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    q = ax.quiver(x0[:, 0], x0[:, 1], x0[:, 2], n0[:, 0],
                  n0[:, 1], n0[:, 2], length=0.1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(title)
    return q

def access_IR(HRTF, measurement, emitter, plot=False):
    '''
    IRs: [2, T]
    relative_loc_sph: [3], azimuth, elevation, distance
    '''
    IRs = []
    N_sample = HRTF.Dimensions.N
    sample_rate = HRTF.Data.SamplingRate.get_values(indices={"M": measurement})
    for receiver in np.arange(HRTF.Dimensions.R):
        IR = HRTF.Data.IR.get_values(indices={"M": measurement, "R": receiver, "E": emitter})
        IRs.append(IR)
    IRs = np.array(IRs)
    relative_loc_sph = HRTF.Emitter.Position.get_relative_values(HRTF.Listener, indices={"M":measurement}, system="spherical", 
                                                                 angle_unit="degree")
    relative_loc_sph = np.round(relative_loc_sph[0], 2)

    # relative_loc_cart = HRTF.Emitter.Position.get_relative_values(HRTF.Listener, indices={"M":measurement}, system="cartesian")
    # relative_loc_cart = np.round(relative_loc_cart[0], 2)
    if plot:
        t = np.arange(0, N_sample) * sample_rate
        plt.figure(figsize=(15, 5))
        plt.plot(t, IRs.T)
        plt.title('HRIR at M={0} for emitter {1}'.format(measurement, emitter))
        plt.xlabel('$t$ in s')
        plt.ylabel(r'$h(t)$')
        plt.grid()
        # plt.savefig('HRTF.png')
    return IRs, relative_loc_sph

class HRTF_simulator():
    def __init__(self, database='ita') -> None:
        if database == 'ita':
            self.init_HRTF('HRTF-Database/ITA HRTF-database/SOFA', user=[0])
            # self.init_HRTF('HRTF-Database/ITA HRTF-database/SOFA', user=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        if database == 'realroom':
            self.init_HRTF('HRTF-Database/RealRoomBRIRs/sofa48k', user=[0])
    def init_HRTF(self, HRTF_folder, user=None):
        self.azimuth_interval = 5
        self.HRTF_folder = HRTF_folder
        self.sofas = os.listdir(self.HRTF_folder)
        self.sofas = [s for s in self.sofas if s.endswith('.sofa')]
        if user is not None:
            self.sofas = [self.sofas[i] for i in user]
        self.HRTFs = []
        for i, s in enumerate(self.sofas):
            HRTF_path = os.path.join(self.HRTF_folder, s)
            HRTF = sofa.Database.open(HRTF_path)
            M = HRTF.Dimensions.M
            HRTF_list = [[]] * (360 // self.azimuth_interval)
            print(f"Loading user {i} with {M} measurements")
            for m in tqdm(range(M)):
                IR, relative_loc_sph = access_IR(HRTF, m, 0)
                azimuth_idx = int(relative_loc_sph[0] // self.azimuth_interval)
                HRTF_list[azimuth_idx].append([relative_loc_sph, IR])
            self.HRTFs.append(HRTF_list)
        self.HRTF = True
    def apply_HRTF(self, signals, doa_degree):
        HRTF_list = sample(self.HRTFs, 1)[0]
        assert len(signals) == len(doa_degree)
        HRTF_signals = []
        for (s, doa) in zip(signals, doa_degree):
            doa_idx = int(doa[0] // self.azimuth_interval)
            IRs = HRTF_list[doa_idx]
            IR = sample(IRs, 1)[0][1]

            left = signal.convolve(s[0], IR[0])
            right = signal.convolve(s[1], IR[1])
            HRTF_signals.append(np.c_[left, right].T)
        HRTF_signals = np.array(HRTF_signals)
        return HRTF_signals  

if __name__ == "__main__":
    hrtf_simulator = HRTF_simulator('realroom')