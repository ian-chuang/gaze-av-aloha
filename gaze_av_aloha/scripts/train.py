import hydra
from gaze_av_aloha.train import train

@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg):
    train(cfg)

if __name__ == "__main__":
    main()