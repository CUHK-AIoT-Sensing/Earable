import argparse
import queue
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from demo import inference_online, init_net

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'channels', type=int, default=[1], nargs='*', metavar='CHANNEL',
    help='input channels to plot (default: the first)')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-w', '--window', type=float, default=1000, metavar='DURATION',
    help='visible time slot (default: %(default)s ms)')
parser.add_argument(
    '-i', '--interval', type=float, default=30,
    help='minimum time between plot updates (default: %(default)s ms)')
parser.add_argument(
    '-b', '--blocksize', type=int, default=320, help='block size (in samples)')
parser.add_argument(
    '-r', '--samplerate', type=float, default=16000, help='sampling rate of audio device')
parser.add_argument(
    '-n', '--downsample', type=int, default=10, metavar='N',
    help='display every Nth sample (default: %(default)s)')
args = parser.parse_args(remaining)
q = queue.Queue()

def callback(indata, frames, time, status):
    global previous_block
    data = np.concatenate((previous_block, indata))
    previous_block = indata
    enhanced = inference_online(data, net)[320:]
    indata = indata[::args.downsample, :1]
    enhanced = enhanced[::args.downsample, np.newaxis]
    q_data = np.concatenate((indata, enhanced), axis=1)
    q.put(q_data)

def update_plot(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines):
        line.set_ydata(plotdata[:, 0])

def update_plot_output(frame):
    """This is called by matplotlib for each plot update.

    Typically, audio callbacks happen more frequently than plot updates,
    therefore the queue tends to contain multiple blocks of audio data.

    """
    global plotdata
    while True:
        try:
            data = q.get_nowait()
        except queue.Empty:
            break
        shift = len(data)
        plotdata = np.roll(plotdata, -shift, axis=0)
        plotdata[-shift:, :] = data
    for column, line in enumerate(lines_output):
        line.set_ydata(plotdata[:, 1])


net = init_net()
previous_block = np.zeros((args.blocksize, 2))
if args.samplerate is None:
    device_info = sd.query_devices(args.device, 'input')
    args.samplerate = device_info['default_samplerate']

length = int(args.window * args.samplerate / (1000 * args.downsample))
plotdata = np.zeros((length, 2))

fig, ax = plt.subplots(1, 2)
lines = ax[0].plot(plotdata[:, 0])
lines_output = ax[1].plot(plotdata[:, 1])
ax[0].set_title('Input')
ax[0].axis((0, len(plotdata), -1, 1))
ax[0].set_yticks([0])
ax[0].yaxis.grid(True)
ax[0].tick_params(bottom=False, top=False, labelbottom=False,
                right=False, left=False, labelleft=False)
ax[1].set_title('Output')
ax[1].axis((0, len(plotdata), -1, 1))
ax[1].set_yticks([0])
ax[1].yaxis.grid(True)
ax[1].tick_params(bottom=False, top=False, labelbottom=False,
                right=False, left=False, labelleft=False)
fig.tight_layout()

stream = sd.InputStream(channels=2, blocksize=args.blocksize,
    samplerate=args.samplerate, callback=callback)
ani1 = FuncAnimation(fig, update_plot, interval=args.interval,)
ani2 = FuncAnimation(fig, update_plot_output, interval=args.interval,)

with stream:
    plt.show()