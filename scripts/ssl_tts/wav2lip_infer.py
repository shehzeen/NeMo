from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, wav2lip_audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform


wav2lip_batch_size = 128
face_det_batch_size = 128
img_size = 96
checkpoint_path = "/data/shehzeen/Wav2LipCkpts/wav2lip.pth"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = face_det_batch_size
    
    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = [0, 10, 0, 0]
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not False: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def datagen(frames, face_det_results, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    
    for i, m in enumerate(mels):
        idx = i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

video_paths = {
    'obama' : "/data/shehzeen/Wav2LipCkpts/obamasamplevideo6.mp4",
}

silent_video_paths = {
    'obama' : "/data/shehzeen/Wav2LipCkpts/obamasilent.mp4",
}

video_frames = {}
video_fps = {}
face_det_results_per_id = {}
silent_video_frames = {}
for person in video_paths:
    video_stream = cv2.VideoCapture(video_paths[person]) 
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    print('Reading video frames...')
    full_frames = []
    while 1:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        
        y1, y2, x1, x2 = [0, -1, 0, -1]
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]

        frame = frame[y1:y2, x1:x2]

        full_frames.append(frame)
    
    video_frames[person] = full_frames
    video_fps[person] = fps
    face_det_results = face_detect(full_frames) # BGR2RGB for CNN face detection
    face_det_results_per_id[person] = face_det_results

    if person in silent_video_paths:
        video_stream = cv2.VideoCapture(silent_video_paths[person]) 
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        print('Reading silent video frames...')
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            
            y1, y2, x1, x2 = [0, -1, 0, -1]
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
        
        silent_video_frames[person] = full_frames

model = load_model(checkpoint_path)

def get_lipsynced_video(wav, person="obama", silence=False):
    if type(wav) == str:
        wav = wav2lip_audio.load_wav(wav, 16000)

    mel = wav2lip_audio.melspectrogram(wav)
    mel_step_size = 16

    mel_chunks = []
    mel_idx_multiplier = 80./video_fps[person] 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    
    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = video_frames[person][:len(mel_chunks)]
    if person in silent_video_frames:
        silent_frames = silent_video_frames[person]
        print("silent frames: {}".format(len(silent_frames)))
        print("full frames: {}".format(len(full_frames)))
    else:
        silent_frames = full_frames
    batch_size = 16
    gen = datagen(full_frames.copy(), face_det_results_per_id[person], mel_chunks)

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.mp4', 
                                    cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        
        idx = 0
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            
            f[y1:y2, x1:x2] = p
            # write frame to jpg
            cv2.imwrite('temp/{}.jpg'.format(idx), f)
            if silence:
                print("writing silent frame")
                out.write(silent_frames[idx % len(silent_frames)])
            else:
                out.write(f)
            idx += 1

    out.release()

    return "temp/result.mp4"

# get_lipsynced_video("/data/shehzeen/SSLTTS/CelebrityData/YoutubeChunkedAudio/emma_1_8.wav", "obama")