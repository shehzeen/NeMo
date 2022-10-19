import torch
import argparse
import json
from pathlib import Path
from nemo.collections.tts.torch.helpers import get_base_dir
import os
import random

class SpeakerClassifer(torch.nn.Module):
    def __init__(self, num_speakers, emb_size=512, dropout=0.1, num_hidden=3):
        super().__init__()
        self.num_speakers = num_speakers
        self.emb_size = emb_size
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(emb_size, 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.linear3 = torch.nn.Linear(256, 256)
        self.linear4 = torch.nn.Linear(256, num_speakers)
        self.loss = torch.nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        x = torch.nn.functional.relu(x)
        x = self.linear3(x)
        x = torch.nn.functional.relu(x)
        x = self.linear4(x)
        
        return x
    
    def get_loss(self, x, y):
        x = self.forward(x)
        loss = self.loss(x, y)
        return loss

class SpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, speaker_records, embedding_type="content"):
        self.embedding_type = embedding_type
        audio_paths = []
        for sample in speaker_records:
            audio_paths.append(sample['audio_filepath'])
        base_dir = get_base_dir(audio_paths)
        sup_data_dir = os.path.join(base_dir, "sup_data")
        print("sup_data_dir", sup_data_dir)
        self.content_embedding_fps = []
        self.speaker_embeddings_fps = []
        speakers = []
        for sample in speaker_records:
            rel_audio_path = Path(sample["audio_filepath"]).relative_to(base_dir).with_suffix("")
            rel_audio_path_as_text_id = str(rel_audio_path).replace("/", "_")
            
            
            content_emb_fn = "embedding_and_probs_content_embedding_{}.pt".format(rel_audio_path_as_text_id)
            speaker_emb_fn = "speaker_embedding_{}.pt".format(rel_audio_path_as_text_id)

            content_emb_path = os.path.join(sup_data_dir, content_emb_fn)
            speaker_emb_path = os.path.join(sup_data_dir, speaker_emb_fn)

            assert os.path.exists(content_emb_path), content_emb_path
            assert os.path.exists(speaker_emb_path), speaker_emb_path

            self.content_embedding_fps.append(content_emb_path)
            self.speaker_embeddings_fps.append(speaker_emb_path)

            speakers.append(sample["speaker"])
        
        self.speakers = speakers

    
    def __len__(self):
        return len(self.content_embedding_fps)
    
    def __getitem__(self, idx):
        if self.embedding_type == "content":
            emb_fp = self.content_embedding_fps[idx]
            content_embeddings = torch.load(emb_fp)
            mean_content_embedding = torch.mean(content_embeddings, dim=1)
            return mean_content_embedding, torch.tensor(self.speakers[idx]).long()
        elif self.embedding_type == "speaker":
            emb_fp = self.speaker_embeddings_fps[idx]
            return torch.load(emb_fp), torch.tensor(self.speakers[idx]).long()

def get_accuracy(model, loader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            correct += torch.sum(y_pred == y)
            total += len(y)
        return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", type=str, default="/data/shehzeen/SSLTTS/manifests/libri_test.json")
    parser.add_argument("--embedding_type", type=str, default="speaker")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speaker_wise_records = {}
    with open(args.manifest_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            record = json.loads(line)
            speaker = record['speaker']
            if speaker not in speaker_wise_records:
                speaker_wise_records[speaker] = []
            speaker_wise_records[speaker].append(record)

    train_records = []
    test_records = []
    for speaker in speaker_wise_records:
        train_records += speaker_wise_records[speaker][:-2]
        test_records += speaker_wise_records[speaker][-2:]

    random.shuffle(train_records)

    train_dataset = SpeakerDataset(train_records, embedding_type=args.embedding_type)
    test_dataset = SpeakerDataset(test_records, embedding_type=args.embedding_type)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_speakers = len(speaker_wise_records)
    embedding_size = 174
    if args.embedding_type == "speaker":
        embedding_size = 256
    
    model = SpeakerClassifer(num_speakers, embedding_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    global_step = 0
    for eooch in range(1000):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            loss = model.get_loss(x, y)
            loss.backward()
            optimizer.step()
            global_step += 1

            
            if global_step % 100 == 0:
                model.eval()
                accuracy = get_accuracy(model, test_loader, device)
                print("Step: {}, Accuracy: {}".format(global_step, accuracy))
                print("Step: {}, Loss: {}".format(global_step, loss.item()))
                print("***" * 10)
                model.train()


if __name__ == "__main__":
    main()