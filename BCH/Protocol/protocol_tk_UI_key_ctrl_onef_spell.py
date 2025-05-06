# import pyaudio
import time
import wave
import os
from random import randint, sample
import random
import time
from datetime import datetime
import numpy as np
import string
from tqdm import tqdm
import sys

from pydub import AudioSegment
from pydub.playback import play, _play_with_simpleaudio

import tkinter as tk
import sounddevice as sd

from enum import Enum
import csv
import threading

from pynput import keyboard

# global vars
def do_nothing():
    return
class status(Enum):
    waiting = 0
    recording = 1
global_status = status.waiting
ROOT_DIR = "BC data"  # where you want to keep the results
# if ROOT_DIR does not exist, create it
if not os.path.exists(ROOT_DIR):
    os.makedirs(ROOT_DIR)




USER_NAME = "test6" #23hlx_word14_wind
USER_INDEX = 1
USER_NAME = str(USER_INDEX) + USER_NAME

# REPEAT_TIMES = 5  # record 26 * REPEAT_TIMES letters
CUHK_NORMAL = ["WORD", "WORD", "WORD","WORD","WORD_NOISE","WORD_ASIDE","EXPRESSION"]
CUHK_EXTEND = ["WORD_40FMCW" for i in range(5)] + ["WORD_HAT", "WORD_MUSIC","WORD_WALK"]
DMU_NORMAL1 = ["WORD", "WORD", "WORD","WORD","WORD_STAND","HEADMOTION"]
DMU_NORMAL2 = ["EXPRESSION","SPELL1", "SPELL2", "SPELL3"]
DMU_EXTEND = ["WORD_40FMCW" for i in range(5)] + ["WORD_NOISE","WORD_HAT", "WORD_MUSIC","WORD_WALK"]
# AUTHORS = ["WORD_250FMCW"]


# ROUND = CUHK_NORMAL
# ROUND = CUHK_EXTEND
# ROUND = DMU_NORMAL1
ROUND = DMU_NORMAL2
# ROUND = DMU_EXTEND
# ROUND = ["

## -------- For real experiment
INDEX = [0,100]
SPELL_INDEX = [0,150]
## -------- For quick test
# INDEX = [0, 10]
# SPELL_INDEX = [0, 10]

MODE = "onef"  # word or facialexp
EXP_REPEAT_TIMES = 5

RECORD_SECONDS = 1.8
SAMPLING_RATE = 48000
RECORD_ON_PC = False
SHUFFLEALL = False


# p = pyaudio.PyAudio()
# info = p.get_host_api_info_by_index(0)
# numdevices = info.get("deviceCount")
input_device_index = 0
    # if the folder does not exist, create it
if not os.path.exists(os.path.join(ROOT_DIR, USER_NAME)):
    os.makedirs(os.path.join(ROOT_DIR, USER_NAME))
    # else if the folder is not empty, quit
elif len(os.listdir(os.path.join(ROOT_DIR, USER_NAME))) > 0:
    print("folder already exists, please change the user name")
    sys.exit()

current_letter = "NONE"
current_letter_index = -1
letters = None
round_index = [0 for i in range(len(ROUND))]

expression = [ "Cheek raiser 笑😊", "Brow raiser 抬眉瞪大眼","brower lower 皱眉😠","Left wink 左闭眼"
                ,"Right wink 右闭眼😉","Blink 双眼闭眼😣","Open Mouth 张大嘴😮", " Mouth 嘟嘴😗","Sneer Left 向左撇嘴🫤"
                ,"Sneer Right 向右撇嘴🫤", "Dimpler 抿嘴😐", "Joy 左歪头张嘴笑(不闭眼)🤣"]
def roundstr(round = 1, round_ele = "None"):
    if round_ele.endswith("NOISE"):
        extraSTR = ", \n 另外，请停止噪音视频!"
    # elif round_ele.endswith("HEADMOTION"):
    #     extraSTR = ", \n 另外，坐在旁边的"
    else:
        extraSTR = ""
        # concatenate the string
    return f"##__{round_ele} Session Ends 🎉🎉🙏🙏 \n 这一轮结束啦，请摘下并重新戴上耳机 \n 请确认超声音频仍在播放，否则请呼叫研究人员 \n 🙏🙏{extraSTR}" # Take off and rewear the earphone to continue \n

def startstr(round_ele = "None"):
    extraSTR = ""
    if round_ele.startswith("WORD") or round_ele.startswith("HEADMOTION"):
        extraSTR += ", \n 请用悄悄话的方式读短语"
    if round_ele.startswith("SPELL"):
        extraSTR += ", \n 请用悄悄话的方式拼出单词的字母，例如：\n apple----(a) (p) (p) (l) (e)，速度稍快也没问题"

    if round_ele.endswith("NOISE"):
        extraSTR += ", \n 另外，请联系研究人员帮忙使用手机在1m处播放噪音视频!!!!!(本轮结束后暂停)"
    elif round_ele.endswith("ASIDE"):
        extraSTR += ", \n 另外，请联系研究人员坐在您的旁边，模拟真实使用环境"
    elif round_ele.endswith("STAND"):
        extraSTR += ", \n 另外，请站起来进行轻读"
    elif round_ele.endswith("THUMBJAW"):
        extraSTR += ", \n 接下来的一轮需要您在念出单词的同时用拇指和下巴做动作"
    elif round_ele.endswith("HAT"):
        extraSTR += ", \n 接下来的一轮需要您戴上鸭舌帽，帽檐朝后"
    elif round_ele.endswith("MUSIC"):
        extraSTR += ", \n 接下来的一轮需要您在播放和音乐混音好的OFDM超声波"
    elif round_ele.endswith("FMCW"):
        extraSTR += ", \n 接下来的一轮需要您在播放FMCW调制的超声波"
    elif round_ele.endswith("WALK"):
        extraSTR += ", \n 接下来的一轮需要您在抱着电脑，在室内行走的同时轻读单词，\n 如有条件请将手机和RODE放置在随身的小挎包中"
    elif round_ele.endswith("HEADMOTION"):
        extraSTR += ", \n 接下来的一轮需要您在念出单词的同时头部朝指示方向转动, \n (⬆) 向上, (⬇) 向下, (-->）向右, (<--) 向左 \n 例如: turn the tide----(⬆) \n 需要您边念边向上自然转动头部"
    elif round_ele.startswith("EXPRESSION"):
        extraSTR = f", \n 表情轮开启，这一轮不读，但需做表情，\n请坚持表情至听到提示音后松开空格键 \n \n \n 我们的表情有: \n" + ", ".join(expression[:3]) + "\n" + ", ".join(expression[3:6]) + "\n" + ", ".join(expression[6:9]) + "\n" + ", ".join(expression[9:12]) #No need to speak. Please Release the space key only after you hear a beep; \n
    else:
        extraSTR = ""
        # concatenate the string
    return f"##__{round_ele} Session Starts 🛫🛫💪💪 \n  新一轮开始啦，请确认超声波音频仍在播放。\n 🙏🙏🙏{extraSTR}" # make sure ultrasonic in ongoing 🛫🛫💪💪 \n


if MODE == "onef":

    # read letter list from file "./Oxford 1000 Export.csv"  row 2-1001, first column
    with open("./Oxford 1035 Export pure_ext.csv", newline="") as spellfile:
        spelldata = list(csv.reader(spellfile))
        spell_list_ori = [row[0] for row in spelldata[1:500]]
        random.seed(USER_INDEX)
        spell_list_ele = sample(spell_list_ori, len(spell_list_ori))
        spell_list_ele = spell_list_ele[SPELL_INDEX[0]:SPELL_INDEX[1]]
        # segment the spell_list_ele into 3 parts
        spell_list_all = [spell_list_ele[:len(spell_list_ele)//3], spell_list_ele[len(spell_list_ele)//3:2*len(spell_list_ele)//3], spell_list_ele[2*len(spell_list_ele)//3:]]
        print(f'length of spell_list_ele: {len(spell_list_ele)}')

    # with open("./Corpus/MS_phrases2.csv", newline="") as csvfile:
    with open("./Oxford_1035_google_ngram_ext.csv", newline="") as csvfile:
        data = list(csv.reader(csvfile))
        letter_list_ori = [row[0] for row in data[1:]]
        # letter_list = letter_list_ori[:260]
        # randomly select 260 words from the list with seed 260
        random.seed(100)
        letter_list_ele = sample(letter_list_ori, len(letter_list_ori))
        letter_list_ele = letter_list_ele[INDEX[0]:INDEX[1]]
        print(len(letter_list_ele))
        # letter_list = letter_list * REPEAT_TIMES

    # letter_list = letter_list_ele + [roundstr(0)]
    letter_list = []
    for i, ROUND_ELE in enumerate(ROUND):
        # if round_ele starts with word, add the word list
        if ROUND_ELE.startswith("WORD"):
            if "40FMCW" in ROUND_ELE:
                ele40 = letter_list_ele[:40]
                letter_list = letter_list + [startstr(ROUND_ELE)] + ele40 + [roundstr(i+1, ROUND_ELE)]
                # letter_list = letter_list + [startstr(ROUND_ELE)] + sample(ele40, len(ele40)) + [roundstr(i+1, ROUND_ELE)]
            else:
                letter_list = letter_list + [startstr(ROUND_ELE)]+ letter_list_ele + [roundstr(i+1, ROUND_ELE)]
                # letter_list = letter_list + [startstr(ROUND_ELE)]+ sample(letter_list_ele, len(letter_list_ele)) + [roundstr(i+1, ROUND_ELE)]
        elif ROUND_ELE == "EXPRESSION":
            letter_list = letter_list + ["##&&" + startstr(ROUND_ELE)] 
            expression_list = expression * EXP_REPEAT_TIMES
            expression_list = sample(expression_list, len(expression_list))
            letter_list = letter_list +  expression_list + [roundstr(i+1, ROUND_ELE)]
        elif ROUND_ELE == "HEADMOTION":
            label_list= ["----(⬆)", "----（⬇）", "----（-->）", "----(<--)"]
            label_list_cat = []
            for label in label_list:
                label_list_cat = label_list_cat + [label] * (len(letter_list_ele) // 4)
            label_list_cat = sample(label_list_cat, len(label_list_cat))
            if len(label_list_cat) < len(letter_list_ele):
                label_list_cat = label_list_cat + label_list[: len(letter_list_ele) - len(label_list_cat)]
            letter_list_headmotion = [letter + label for letter, label in zip(letter_list_ele, label_list_cat)]
            letter_list = letter_list + [startstr(ROUND_ELE)] 
            letter_list = letter_list + letter_list_headmotion + [roundstr(i+1, ROUND_ELE)]

        elif ROUND_ELE.startswith("SPELL"):
            spellround = int(ROUND_ELE[-1])
            letter_list = letter_list + [startstr(ROUND_ELE)] 
            letter_list = letter_list + spell_list_all[spellround-1] + [roundstr(i+1, ROUND_ELE)] 

        round_index[i] = len(letter_list) # record the end index of each round
        

# elif MODE == "sentence":
#     letter_list = ["test", "test2", "test3"]

# elif MODE == "letter":
#     letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"
#                 , "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w"
#                 , "x", "y", "z"]
#     letter_list = letters * REPEAT_TIMES
# elif MODE == "facialexp":
#     letter_list = expression * EXP_REPEAT_TIMES
#     letter_list = sample(letter_list, len(letter_list))
#     letter_list = [letter + " (> 2s)" for letter in letter_list]


else:
    print("wrong mode")

letter_list = sample(letter_list, len(letter_list)) if SHUFFLEALL else letter_list

# adding the movement label
# adding label of "___(⬆)","___（⬇）","___（-->）","___(<--)" to the end of each word. Each label will be add to consective words of roughly a quater of the list.


# # for demonstration
# letter_list = ["REHEARSE"]

# letter_list.append("BLANK")
recording_thread = None
current_file_name = None

csv_data = []


# Index to Round Number
def index_to_round(index, round_index = round_index):
    for i, round_end in enumerate(round_index):
        if index < round_end:
            return i
    return


# detection_signal = AudioSegment.from_wav("48000_18k20k_005.wav")
def generate_start_tone(freq = 2000, duration = 0.5, volume_adjust = 0.25):
    signal = (
        np.sin(
            2
            * np.pi
            * freq
            * np.linspace(0, duration, int(SAMPLING_RATE * duration), endpoint=False)
        )
        * volume_adjust
    )
    return signal


start_tone_played = False

start_tone = generate_start_tone()
remind_tone = generate_start_tone(500, 0.2, 0.1)
timer_flag = False

global_tag = []


def mark():
    global global_tag
    if len(global_tag) == 0:
        print("wrong")
        return
    global_tag.append(time.monotonic())


def record_file(filename):
    if RECORD_ON_PC:
        CHUNK = 1024
        # FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = SAMPLING_RATE
        global input_device_index

        WAVE_OUTPUT_FILENAME = filename
        # input("press enter to start")
        # p = pyaudio.PyAudio()

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=input_device_index,
        )
        print("start")
        # _play_with_simpleaudio(detection_signal)
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        # input("finished rescording, press ENTER to continue to the next letter")
        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        wf.close()
    else:
        time.sleep(RECORD_SECONDS)


def record_a_letter(user_root_dir, letter):
    global current_file_name
    # letter = letters[randint(0,25)]
    # identifier = str(int(time.time()))
    identifier = time.strftime("%Y-%m-%d  %H:%M:%S", time.localtime())
    # print(letter)
    current_file_name = os.path.join(user_root_dir, letter, identifier) + ".wav"
    record_file(current_file_name)

    # os.system(clear_cmd)


# tk gui
def record():
    global record_flag
    global current_letter
    global recording_thread
    global global_tag

    print(current_letter)
    global_tag = [current_letter]
    mark()
    start_indicater.config(text="start", bg="green")
    record_flag = True
    record_button.config(command=do_nothing)
    retake_button.config(command=do_nothing)
    recording_thread = threading.Thread(
        target=record_a_letter, args=(os.path.join(ROOT_DIR, USER_NAME), current_letter)
    )
    recording_thread.start()
    # record_a_letter(ROOT_DIR, current_letter)

    # update()


def retake_last_letter():
    global current_file_name
    global recording_thread
    global record_flag
    global current_letter_index
    global current_letter
    global timer_flag


    if current_file_name is not None and RECORD_ON_PC:
        os.remove(current_file_name)
    if recording_thread is not None:
        recording_thread.join()
    if current_letter_index > 0:
        current_letter_index = current_letter_index - 1
    else:
        current_letter_index = 0
    if len(csv_data) > 0:
        csv_data.pop()
    current_letter = letter_list[current_letter_index]
    # letter_label.config(text=current_letter)
    # update font and text
    
    if not current_letter.startswith("##&&"):
        timer_flag = False # reset the timer flag if not expression round
    if current_letter.startswith("##"):          
        letter_label.config(text=current_letter, font=("Arial", 30),wraplength=1200)
    else:
        letter_label.config(text=current_letter, font=("Arial Rounded MT Bold", 45),wraplength=1200)
    # progress_indicater.config(text=f"Progress: {current_letter_index+1}/{len(letter_list)}")
    round_num = index_to_round(current_letter_index)
    round_name = ROUND[round_num]
    progress_indicater.config(text=f"Progress: Session {round_name} {round_num+1}/{len(ROUND)}, Sample {current_letter_index+1}/{len(letter_list)}") 
    record_button.config(text="record", command=record)


def update_gui():
    global current_letter


def end():
    global csv_data
    letter_label.config(text="🎉🎉🎉Finished! Thank you so much! \n 这一轮实验告一段落。非常感谢！麻烦稍后填写一个问卷")

    # write csv
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d %H_%M_%S")
    with open(
        os.path.join(ROOT_DIR, USER_NAME, USER_NAME + dt_string + ".csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    record_button.config(command=do_nothing)

    # root.destroy()

def save_csv():
    global csv_data
    # write csv
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d %H_%M_%S")
    with open(
        os.path.join(ROOT_DIR, USER_NAME, USER_NAME + dt_string + ".csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)


def next_letter():
    global recording_thread
    global record_flag
    global current_letter_index
    global current_letter
    global global_tag
    global start_tone_played
    global timer_flag
    global remind_tone

    if not start_tone_played:
        global_tag = ["start"]
        mark()

        sd.play(start_tone, SAMPLING_RATE)
        mark()

        csv_data.append(global_tag[:])

        start_tone_played = True
    if recording_thread is not None:
        recording_thread.join()
    if current_letter_index < len(letter_list):
        current_letter_index += 1

    if current_letter_index == len(letter_list):
        print("end")
        end()
        # return
    else:
        print(current_letter_index, len(letter_list))
        current_letter = letter_list[current_letter_index]
        # if current letter contains &&, it is a special signal
        if current_letter.startswith("##&&"):
            timer_flag = True
            update_timer()
        else:
            timer_flag = False
        
        if current_letter.startswith("##__SPELL"):
            remind_tone = np.zeros(1)
        if current_letter.startswith("##"):          
            letter_label.config(text=current_letter, font=("Arial", 30),wraplength=1200)
        else:
            letter_label.config(text=current_letter, font=("Arial Rounded MT Bold", 45),wraplength=1200)

        record_button.config(text="record", command=record)
        retake_button.config(command=retake_last_letter)
        # progress_indicater.config(text=f"Progress: {current_letter_index+1}/{len(letter_list)}")
        round_num = index_to_round(current_letter_index)
        round_name = ROUND[round_num]
        progress_indicater.config(text=f"Progress: Session {round_name} {round_num+1}/{len(ROUND)}, Sample {current_letter_index+1}/{len(letter_list)}") 
         

    # print(csv_data)


def stop():
    global record_flag
    record_flag = False
    global time_passed
    time_passed = 0

    start_indicater.config(text="stop", bg="red")


def update_timer():
    global record_flag
    global time_passed
    global global_tag
    global timer_flag

    if record_flag: #and timer_flag
        time_passed += 0.1
        if time_passed > 1.95 and time_passed < 2.05:
            sd.play(remind_tone, SAMPLING_RATE)
        # Used for 2s standard recording
        # if time_passed >= 2:
        #     stop()
        #     mark()

        #     csv_data.append(global_tag[:])

        #     next_letter()

        # start_indicater.config(text="start", bg="green")
    else:
        time_passed = 0

    time_passed_label.config(text=format(time_passed, ".1f"))
    root.after(100, update_timer)


# make dirs

all_letters = letter_list
if RECORD_ON_PC:
    for letter in all_letters:
        if not os.path.exists(os.path.join(ROOT_DIR, USER_NAME, letter)):
            os.makedirs(os.path.join(ROOT_DIR, USER_NAME, letter))

#

root = tk.Tk()
root.title = "BCH Data Collection"
root.geometry("1200x800")
# root.configure(background="black")
record_flag = False
time_passed = 0
letter_label = tk.Label(root, text=f"get ready 如有需要可以先上厕所 \n 您当前的轮次是 {ROUND}", fg="green",wraplength=1200)
letter_label.config(font=("Arial Rounded MT Bold", 26))

letter_label_tips = tk.Label(root, text="Tips: \n If you feel like swallowing, please wait until the key is released")
letter_label_tips.config(font=("Arial Rounded MT Bold", 26))

progress_indicater = tk.Label(root, text=f"Progress: 0/{len(letter_list)}")
progress_indicater.config(font=("Arial", 26))

record_button = tk.Button(root, text="start", command=next_letter)
record_button.config(font=("Arial", 26))
retake_button = tk.Button(root, text="Redo Last One" , command=do_nothing) #+ MODE.upper()
save_button = tk.Button(root, text="Save Current CSV", command=save_csv)
time_passed_label = tk.Label(root, text="0.0")
start_indicater = tk.Label(root, text="wait", bg="red")
start_indicater.config(font=("Arial Rounded MT Bold", 26))

record_button.pack(pady=20)
start_indicater.pack(pady=20)
letter_label.pack(pady=20)
# stop_button.pack()
time_passed_label.pack(pady=20)
progress_indicater.pack(pady=20)
retake_button.pack(pady=20)
letter_label_tips.pack(pady=20)
save_button.pack(pady=20)


# for letter in letter_list:
#     current_letter = letter
#     letter_label.config(text=letter)
#     record_button.config(command=record)
#     start_indicater.config(text="wait")


space_pressed = False


def on_key_pressed(key):
    global space_pressed

    if not space_pressed and key == keyboard.Key.space:
        space_pressed = True
        print("pressed")

        global record_flag
        global current_letter
        global recording_thread
        global global_tag

        print(current_letter)
        global_tag = [current_letter]
        mark()
        start_indicater.config(text="start", bg="green")
        record_flag = True
        record_button.config(command=do_nothing)
        retake_button.config(command=do_nothing)
        # recording_thread = threading.Thread(
        #     target=record_a_letter, args=(os.path.join(ROOT_DIR, USER_NAME), current_letter)
        # )
        # recording_thread.start()


def on_key_released(key):
    global space_pressed

    if space_pressed and key == keyboard.Key.space:
        mark()
        stop()
        csv_data.append(global_tag[:])
        space_pressed = False
        start_indicater.config(text="stop", bg="red")
        print("released")

        next_letter()


listener = keyboard.Listener(on_press=on_key_pressed, on_release=on_key_released)
listener.start()


# update_timer() if MODE == "facialexp" else None
root.mainloop()
# tkgui ends
