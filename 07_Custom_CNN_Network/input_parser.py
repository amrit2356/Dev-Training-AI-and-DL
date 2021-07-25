import argparse
import torch
import multiprocessing


def parse_input():
    parser = argparse.ArgumentParser()
    # Project File Location Inputs(Project Name, Dataset Path, Output Path->train folder, checkpoints and Exported Model)
    parser.add_argument(
        '--project_name', 
        type=str, 
        required=True, 
        help='Enter the Project Name'
    )
    parser.add_argument(
        '--dataset_path', 
        type=str, 
        required=True, 
        help='Entert the Dataset Path'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help='Enter the Project Folder Path to store checkpoints, exports and train file'
    )
    
    # Data Type Input(Torchvision Dataset, Custom)
    parser.add_argument(
        '--data_type', 
        type=str, 
        required=True, 
        help='Type of Data to train(Torchvision Datasets(CIFAR10)/custom)'
    )
    
    # Training Parameters Input(Batch Size, Learning Rate, Num of classes to classify)
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=512, 
        help='Set the batch size for Training'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.0001, 
        help="Set the Learning Rate for Training"
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=5, 
        help='Set the number of epochs for Training'
    )
    parser.add_argument(
        '--num_classes', 
        type=int, 
        required=True, 
        help='Set the number of classes to classify for Training'
    )

    # Normalization Parameters Inputs
    parser.add_argument(
        "--num-samples",
        metavar="N",
        type=int,
        default=None,
        help="Number of images used in the calculation. Defaults to the complete dataset.",
    )
    parser.add_argument(
        "--num-workers",
        metavar="N",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of workers for the image loading. Defaults to the number of CPUs.",
    )
    parser.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=None,
        help="Number of images processed in parallel. Defaults to the number of workers",
    )
    parser.add_argument(
        "--device",
        metavar="DEV",
        type=str,
        default=None,
        help="Device to use for processing. Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--seed",
        metavar="S",
        type=int,
        default=None,
        help="If given, runs the calculation in deterministic mode with manual seed S.",
    )
    parser.add_argument(
        "--print_freq",
        metavar="F",
        type=int,
        default=50,
        help="Frequency with which the intermediate results are printed. Defaults to 50.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If given, only the final results is printed",
    )

    args = parser.parse_args()

    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()

    if args.batch_size is None:
        args.batch_size = args.num_workers

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(device)


    args = parser.parse_args()
    return args
