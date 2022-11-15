import torch
from torch.utils.tensorboard import SummaryWriter
import yaml
from model import DDSP
from effortless_config import Config
from os import path
from preprocess import Dataset
from tqdm import tqdm
from core import multiscale_fft, safe_log, mean_std_loudness
import soundfile as sf
from einops import rearrange
from utils import get_scheduler
import numpy as np
import os
import customPath
from scipy.io.wavfile import write
import logging

class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    ROOT = "runs"
    STEPS = 500000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 400000
    DATASET = "synthetic"

args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

torch.manual_seed(4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(path.join(customPath.models(), args.NAME), exist_ok=True)
logging.basicConfig(filename=os.path.join(customPath.models(), args.NAME, 'training.log'), level=logging.INFO, format='%(name)s - %(asctime)s - %(message)s')

model = DDSP(**config["model"]).to(device)

if args.DATASET == "synthetic":
    dataset_train = Dataset(os.path.join(customPath.synthetic(), 'preprocessed/train'))
    dataset_test = Dataset(os.path.join(customPath.synthetic(), 'preprocessed/test'))
elif args.DATASET == "sol":
    dataset_train = Dataset(os.path.join(customPath.orchideaSOL(), 'preprocessed/train'))
    dataset_test = Dataset(os.path.join(customPath.orchideaSOL(), 'preprocessed/test'))

dataloader = torch.utils.data.DataLoader(
    dataset_train,
    args.BATCH,
    True,
    drop_last=True,
)

mean_loudness, std_loudness = mean_std_loudness(dataloader)
config["data"]["mean_loudness"] = mean_loudness
config["data"]["std_loudness"] = std_loudness

writer = SummaryWriter(path.join(customPath.models(), args.NAME), flush_secs=20)

with open(path.join(customPath.models(), args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)

schedule = get_scheduler(
    len(dataloader),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

# scheduler = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
# scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10000, gamma=0.1, verbose=True)

best_loss = float("inf")
mean_loss = 0
n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader)))

for e in tqdm(range(epochs)):
    index_s = 0
    for s_WB, s_LB, p, l in dataloader:
        # s = s/s.max()
        s_WB = s_WB.to(device)
        s_LB = s_LB.to(device)
        p = p.unsqueeze(-1).to(device)
        l = l.unsqueeze(-1).to(device)

        l = (l - mean_loudness) / std_loudness

        if config['data']['input'] == 'LB':
            y = model(s_LB, p, l).squeeze(-1)
        elif config['data']['input'] == 'WB':
            y = model(s_WB, p, l).squeeze(-1)

        ori_stft = multiscale_fft(
            s_WB,
            config["train"]["scales"],
            config["train"]["overlap"],
        )
        rec_stft = multiscale_fft(
            y,
            config["train"]["scales"],
            config["train"]["overlap"],
        )

        loss = 0
        for s_x, s_y in zip(ori_stft, rec_stft):
            lin_loss = (s_x - s_y).abs().mean()
            log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
            loss = loss + lin_loss + log_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        writer.add_scalar("loss", loss.item(), step)

        step += 1

        n_element += 1
        mean_loss += (loss.item() - mean_loss) / n_element
        logging.info(f'Step {step}, loss: {mean_loss}')


    if not e % 10:
        writer.add_scalar("lr", schedule(e), e)
        writer.add_scalar("reverb_decay", model.reverb.decay.item(), e)
        writer.add_scalar("reverb_wet", model.reverb.wet.item(), e)
        # scheduler.step()
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                model.state_dict(),
                path.join(customPath.models(), args.NAME, "state.pth"),
            )

        mean_loss = 0
        n_element = 0     

        index_s += 1

logging.info("\nTraining done.")
logging.info("Generating some training data ...")

# generate some examples from train dataset
dataloader = torch.utils.data.DataLoader(
    dataset_train,
    1,
    False,
    drop_last=True,
)
model.load_state_dict(torch.load(os.path.join(customPath.models(), args.NAME, "state.pth"), map_location=torch.device('cpu')))
model.eval()

index_s = 0
for s_WB, s_LB, p, l in dataloader:
    # s = s/s.max()
    s_WB = s_WB.to(device)
    s_LB = s_LB.to(device)
    p = p.unsqueeze(-1).to(device)
    l = l.unsqueeze(-1).to(device)

    l = (l - mean_loudness) / std_loudness
    if config['data']['input'] == 'LB':
        y = model(s_LB, p, l).squeeze(-1)
    elif config['data']['input'] == 'WB':
        y = model(s_WB, p, l).squeeze(-1)

    if not index_s % 250:
        orig_audio = s_WB[0].detach().cpu().numpy()
        regen_audio = y[0].detach().cpu().numpy()
        sf.write(
            path.join(customPath.models(), args.NAME, f"orig_{index_s}.wav"),
            orig_audio,
            config["preprocess"]["sampling_rate"],
        )
        sf.write(
            path.join(customPath.models(), args.NAME, f"regen_{index_s}.wav"),
            regen_audio,
            config["preprocess"]["sampling_rate"],
        )

    index_s += 1

logging.info("Training data reconstructions generated.")

# generate some examples from train dataset
dataloader = torch.utils.data.DataLoader(
    dataset_test,
    1,
    False,
    drop_last=True,
)

index_s = 0
for s_WB, s_LB, p, l in dataloader:
    # s = s/s.max()
    s_WB = s_WB.to(device)
    s_LB = s_LB.to(device)
    p = p.unsqueeze(-1).to(device)
    l = l.unsqueeze(-1).to(device)

    l = (l - mean_loudness) / std_loudness
    if config['data']['input'] == 'LB':
        y = model(s_LB, p, l).squeeze(-1)
    elif config['data']['input'] == 'WB':
        y = model(s_WB, p, l).squeeze(-1)

    if not index_s % 250:
        orig_audio = s_WB[0].detach().cpu().numpy()
        regen_audio = y[0].detach().cpu().numpy()
        sf.write(
            path.join(customPath.models(), args.NAME, f"orig_{index_s}.wav"),
            orig_audio,
            config["preprocess"]["sampling_rate"],
        )
        sf.write(
            path.join(customPath.models(), args.NAME, f"regen_{index_s}.wav"),
            regen_audio,
            config["preprocess"]["sampling_rate"],
        )

    index_s += 1

logging.info("Training data reconstructions generated.")