import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Draw figure to plot the loss and accuracy
import matplotlib.pyplot as plt

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from sinkhorn_models_num_bucket import SinkhornViTBinaryClassifier
from sinkhorn_models_diff_len_sub import SinkhornViTBinaryClassifier
from torch_utils import set_seed, get_device, numpy_to_data_loader, model_fit, classification_acc

import wandb
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SinkhornViTBinaryClassifier.")
    
    parser.add_argument('--num_buckets', type=int, default=8, help="Number of buckets")
    parser.add_argument('--depth', type=int, default=2, help="Depth of the model")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=30, help="Number of epochs")
    parser.add_argument('--temperature', type=float, default=0.1, help="Temperature")
    parser.add_argument('--iterations', type=int, default=5, help="Number of Sinkhorn iterations")
    
    return parser.parse_args()

def collate_fn(batch_of_pairs, bucket_size=None, sub_len=100):
    # Step 1: Determine the maximum sequence length in the batch
    max_length = max(max(len(pair[0][0]), len(pair[0][1])) for pair in batch_of_pairs)
    print("max_length", max_length)

    # If bucket_size is provided, ensure max_length is a multiple of bucket_size
    if bucket_size is not None and max_length % bucket_size != 0:
        max_length = ((max_length // bucket_size) + 1) * bucket_size

    # Step 2: Padding each trajectory in the pair
    padded_pairs = []
    labels = []
    for pair in batch_of_pairs:
        padded_trajectory_1 = np.pad(pair[0][0], ((0, max_length - len(pair[0][0])), (0, 0)), mode='constant', constant_values=0)
        padded_trajectory_2 = np.pad(pair[0][1], ((0, max_length - len(pair[0][1])), (0, 0)), mode='constant', constant_values=0)
        padded_pairs.append([padded_trajectory_1, padded_trajectory_2])

        # Calculate the number of sub-trajectories
        num_sub_trajectories = max_length // sub_len

        # Generate labels based on the custom scheme
        num_positive_labels = (num_sub_trajectories * (num_sub_trajectories - 1)) // 2
        num_negative_labels = num_sub_trajectories * num_sub_trajectories

        # Generate positive labels (1) for the upper triangular portion of the matrix
        positive_labels = np.ones(num_positive_labels)

        # Generate negative labels (0) for the lower triangular portion of the matrix
        negative_labels = np.zeros(num_negative_labels)

        # Concatenate the labels for the batch
        batch_labels = np.concatenate((positive_labels, negative_labels, positive_labels))
        labels.append(batch_labels)

    # Step 3: Form the Batch of Pairs
    padded_batch = torch.tensor(padded_pairs, dtype=torch.float32)

    # Step 4: Normalize the entire padded batch
    padded_batch[:, :, :, 0] /= 92
    padded_batch[:, :, :, 1] /= 49
    padded_batch[:, :, :, 2] /= 288
    print("a", padded_batch.shape)
    labels = torch.tensor(labels).float()  # Convert labels to float
    print("b", labels.shape)

    return padded_batch, labels

def main(args):
    wandb.init(
    project="my-awesome-project",
    config=vars(args)
    )


    num_buckets = args.num_buckets
    depth = args.depth
    learning_rate = args.learning_rate
    epochs = args.epochs
    temperature = args.temperature
    iterations = args.iterations
    base_filename = (f"th1_test_different_num_buckets_sinkhorn5_30epochs_bucket_"
                 f"{args.num_buckets}_depth_{args.depth}_tem_{args.temperature:.3f}_iter_{args.iterations}_weighted_sum")

    model_save_path = f"final_final/{base_filename}.pth"
    plot_save_path = f"final_final/{base_filename}.png"

    device = get_device()

    # Load pre-processed data and convert to PyTorch tensors
    x_train, y_train = pickle.load(open("whole_day_status_with_speed/siamese_train_class_whole_day_with_status_070809_speed_100_50000.pkl", "rb"))
    x_val, y_val = pickle.load(open("whole_day_status_with_speed/siamese_val_class_whole_day_with_status_070809_speed_100_10000.pkl", "rb"))

    train_data = list(zip(x_train, y_train))
    val_data = list(zip(x_val, y_val))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=lambda x: collate_fn(x, bucket_size=None))
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False, collate_fn=lambda x: collate_fn(x, bucket_size=None))

    model = SinkhornViTBinaryClassifier(
        sub_len = 100,
        num_tokens=5,
        dim=128, 
        depth=depth,
        local_window_size = 0,
        heads = 8,
        temperature=temperature,
        num_buckets = num_buckets,
        sinkhorn_iter = iterations,
        attn_dropout = 0.1,
        n_local_attn_heads = 0
        )
    
    # Check the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of available GPUs:", num_gpus)

    if num_gpus > 1:
        # Wrap the model with nn.DataParallel
        model = nn.DataParallel(model)
    
    # Train model
    model.to(device)
    loss_fn = nn.BCELoss()
    acc_fn = classification_acc("binary")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    EPOCHS = epochs

    history = model_fit(
        model,
        loss_fn,
        acc_fn,
        optimizer,
        train_loader,
        epochs=EPOCHS,
        val_loader=val_loader,
        save_best_only=True,
        early_stopping=30,
        save_every_epoch=True,
        save_path=model_save_path,
        # save_path=f'wandb_speed_final/model_different_num_buckets_sinkhorn_Siamese_with_status_July_100_20000_30epochs_bucket_{num_buckets}_depth_{depth}_temperature_025_nt.pth',
        device=device,
    )


    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']

    # calculate total parameters in model
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')

    EPOCHS = len(train_loss)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(EPOCHS), train_loss, label='train')
    plt.plot(range(EPOCHS), val_loss, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(EPOCHS), train_acc, label='train')
    plt.plot(range(EPOCHS), val_acc, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.savefig(plot_save_path)
    # plt.savefig(f'wandb_speed_final/model_different_num_buckets_sinkhorn_Siamese_with_status_July_100_20000_30epochs_bucket_{num_buckets}_depth_{depth}_temperature_025_nt.png')

if __name__ == '__main__':
    args = parse_args()
    main(args)