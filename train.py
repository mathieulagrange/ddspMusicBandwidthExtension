import torch
import yaml
from model_ddsp import DDSP
from model_resnet import Resnet
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

generate_examples = False

class args(Config):
    CONFIG = "config.yaml"
    NAME = "debug"
    STEPS = 25000
    BATCH = 16
    START_LR = 1e-3
    STOP_LR = 1e-4
    DECAY_OVER = 5000
    DATASET = "synthetic"

args.parse_args()

with open(args.CONFIG, "r") as config:
    config = yaml.safe_load(config)

torch.manual_seed(4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(path.join(customPath.models(), args.NAME), exist_ok=True)
logging.basicConfig(filename=os.path.join(customPath.models(), args.NAME, 'training.log'), level=logging.INFO, format='%(name)s - %(asctime)s - %(message)s')

if config['preprocess']['sampling_rate'] == 16000:
    if config['train']['model'] == 'ddsp':
        if args.DATASET == "synthetic":
            dataset_train = Dataset(os.path.join(customPath.synthetic(), 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.synthetic(), 'preprocessed_ddsp/test'), model='ddsp')
        elif args.DATASET == "sol":
            dataset_train = Dataset(os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/test'), model='ddsp')
        elif args.DATASET == "medley":
            dataset_train = Dataset(os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.medleySolosDB(), 'preprocessed_ddsp/test'), model='ddsp')
        elif args.DATASET == "dsd_sources":
            dataset_train = Dataset(os.path.join(customPath.dsd_sources(), 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.dsd_sources(), 'preprocessed_ddsp/test'), model='ddsp')
        elif args.DATASET == "dsd_mixtures":
            dataset_train = Dataset(os.path.join(customPath.dsd_mixtures(), 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.dsd_mixtures(), 'preprocessed_ddsp/test'), model='ddsp')

        model = DDSP(**config["model"]).to(device)

    elif config['train']['model'] == 'resnet':
        if args.DATASET == "synthetic":
            dataset_train = Dataset(os.path.join(customPath.synthetic(), 'preprocessed_resnet/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.synthetic(), 'preprocessed_resnet/test'), model='resnet')
        elif args.DATASET == "sol":
            dataset_train = Dataset(os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/test'), model='resnet')
        elif args.DATASET == "medley":
            dataset_train = Dataset(os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet/test'), model='resnet')
        elif args.DATASET == "dsd_sources":
            dataset_train = Dataset(os.path.join(customPath.dsd_sources(), 'preprocessed_resnet/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.dsd_sources(), 'preprocessed_resnet/test'), model='resnet')
        elif args.DATASET == "dsd_mixtures":
            dataset_train = Dataset(os.path.join(customPath.dsd_mixtures(), 'preprocessed_resnet/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.dsd_mixtures(), 'preprocessed_resnet/test'), model='resnet')

        model = Resnet().to(device)

elif config['preprocess']['sampling_rate'] == 8000:
    if config['train']['model'] == 'ddsp':
        if args.DATASET == "synthetic":
            dataset_train = Dataset(os.path.join(customPath.synthetic(), '8000', 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.synthetic(), '8000', 'preprocessed_ddsp/test'), model='ddsp')
        elif args.DATASET == "sol":
            dataset_train = Dataset(os.path.join(customPath.orchideaSOL(), '8000', 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.orchideaSOL(), '8000', 'preprocessed_ddsp/test'), model='ddsp')
        elif args.DATASET == "medley":
            dataset_train = Dataset(os.path.join(customPath.medleySolosDB(), '8000', 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.medleySolosDB(), '8000', 'preprocessed_ddsp/test'), model='ddsp')
        elif args.DATASET == "dsd_sources":
            dataset_train = Dataset(os.path.join(customPath.dsd_sources(), '8000', 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.dsd_sources(), '8000', 'preprocessed_ddsp/test'), model='ddsp')
        elif args.DATASET == "dsd_mixtures":
            dataset_train = Dataset(os.path.join(customPath.dsd_mixtures(), '8000', 'preprocessed_ddsp/train'), model='ddsp')
            dataset_test = Dataset(os.path.join(customPath.dsd_mixtures(), '8000', 'preprocessed_ddsp/test'), model='ddsp')

        model = DDSP(**config["model"]).to(device)

    elif config['train']['model'] == 'resnet':
        if args.DATASET == "synthetic":
            dataset_train = Dataset(os.path.join(customPath.synthetic(), 'preprocessed_resnet/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.synthetic(), 'preprocessed_resnet/test'), model='resnet')
        elif args.DATASET == "sol":
            dataset_train = Dataset(os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.orchideaSOL(), 'preprocessed_ddsp/test'), model='resnet')
        elif args.DATASET == "medley":
            dataset_train = Dataset(os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.medleySolosDB(), 'preprocessed_resnet/test'), model='resnet')
        elif args.DATASET == "dsd_sources":
            dataset_train = Dataset(os.path.join(customPath.dsd_sources(), 'preprocessed_resnet/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.dsd_sources(), 'preprocessed_resnet/test'), model='resnet')
        elif args.DATASET == "dsd_mixtures":
            dataset_train = Dataset(os.path.join(customPath.dsd_mixtures(), 'preprocessed_resnet/train'), model='resnet')
            dataset_test = Dataset(os.path.join(customPath.dsd_mixtures(), 'preprocessed_resnet/test'), model='resnet')


dataloader = torch.utils.data.DataLoader(
    dataset_train,
    args.BATCH,
    True,
    drop_last=True,
)

if config['train']['model'] == 'ddsp':
    mean_loudness, std_loudness = mean_std_loudness(dataloader)
    config["data"]["mean_loudness"] = mean_loudness
    config["data"]["std_loudness"] = std_loudness

with open(path.join(customPath.models(), args.NAME, "config.yaml"), "w") as out_config:
    yaml.safe_dump(config, out_config)

if config['train']['model'] == 'ddsp':
    opt = torch.optim.Adam(model.parameters(), lr=args.START_LR)
elif config['train']['model'] == 'resnet':
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)

schedule = get_scheduler(
    len(dataloader),
    args.START_LR,
    args.STOP_LR,
    args.DECAY_OVER,
)

best_loss = float("inf")
mean_loss = 0

n_element = 0
step = 0
epochs = int(np.ceil(args.STEPS / len(dataloader)))

# LR decreased if plateau
mean_loss_lr_plateau = 0
count_plateau = 0
last_best_mean_loss = 0
first_check_plateau = True

for e in tqdm(range(epochs)):
    index_s = 0
    for batch in dataloader:
        if config['train']['model'] == 'ddsp':
            s_WB, s_LB, p, l = batch
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
                HF = config['train']['HF']
            )
            rec_stft = multiscale_fft(
                y,
                config["train"]["scales"],
                config["train"]["overlap"],
                HF = config['train']['HF']
            )

            loss = 0
            for s_x, s_y in zip(ori_stft, rec_stft):
                lin_loss = (s_x - s_y).abs().mean()
                log_loss = (safe_log(s_x) - safe_log(s_y)).abs().mean()
                loss = loss + lin_loss + log_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            n_element += 1
            mean_loss += (loss.item() - mean_loss) / n_element

        elif config['train']['model'] == 'resnet':
            s_WB, s_LB = batch
            s_WB = s_WB.unsqueeze(1).to(device)
            s_LB = s_LB.unsqueeze(1).to(device)
            
            y = model(s_LB)

            mse_loss = torch.nn.MSELoss().to(device)(y, s_WB)

            opt.zero_grad()
            mse_loss.backward()
            opt.step()
            step += 1
            n_element += 1
            mean_loss = (mean_loss*(n_element-1)/n_element) + mse_loss/(n_element)
            mean_loss_lr_plateau = (mean_loss_lr_plateau*(n_element-1)/n_element) + mse_loss/(n_element)

        logging.info(f'Step {step}, loss: {mean_loss}')

        if not step % 2500:
            if first_check_plateau and mean_loss_lr_plateau != 0.0:
                last_best_mean_loss = mean_loss_lr_plateau
                first_check_plateau = False
            else:
                if mean_loss_lr_plateau < last_best_mean_loss:
                    count_plateau = 0
                    last_best_mean_loss = mean_loss_lr_plateau
                else:
                    count_plateau += 1
            
            if count_plateau == 5:
                count_plateau = 0
                opt.param_groups[0]['lr'] = opt.param_groups[0]['lr']/2
                lr = opt.param_groups[0]['lr']
                logging.info(f'Learning rate halved to {lr}')

            mean_loss_lr_plateau = 0
            
    if not e % 10:
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

### EXAMPLES GENERATION ###
if generate_examples:
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
                path.join(customPath.models(), args.NAME, f"train_orig_{index_s}.wav"),
                orig_audio,
                config["preprocess"]["sampling_rate"],
            )
            sf.write(
                path.join(customPath.models(), args.NAME, f"train_regen_{index_s}.wav"),
                regen_audio,
                config["preprocess"]["sampling_rate"],
            )

        index_s += 1

    logging.info("Training data reconstructions generated.")

    # generate some examples from test dataset
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
                path.join(customPath.models(), args.NAME, f"test_orig_{index_s}.wav"),
                orig_audio,
                config["preprocess"]["sampling_rate"],
            )
            sf.write(
                path.join(customPath.models(), args.NAME, f"test_regen_{index_s}.wav"),
                regen_audio,
                config["preprocess"]["sampling_rate"],
            )

        index_s += 1

    logging.info("Testing data reconstructions generated.")