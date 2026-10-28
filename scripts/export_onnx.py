import argparse
import torch
from src.training.learner import RecurrentLearner
from src.deployment.onnx_policy import export_actor

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();torch.set_num_threads(1)
    model=RecurrentLearner.from_checkpoint(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    print(export_actor(model,args.output))
if __name__=='__main__':main()
