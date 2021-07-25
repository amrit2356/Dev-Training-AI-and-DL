import argparse

import numpy as np
import torch

from models.team_classifier_cnn import TeamClassifier as Model


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(num_classes=2).to(device)
    model.eval()

    x = torch.rand(args.batch_size, 3, 160, 64).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))

    for i in range(10):
        _ = model(x)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(x)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.mean(timings)
    print('Mean time: {:.2f}ms per batch'.format(mean_syn))
    print('Inference time per image: {:.2f}μs'.format(mean_syn * 1000 / args.batch_size))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--batch_size', type=int, default=1024, help='Number of images to run inference on')

    main(argparser.parse_args())
