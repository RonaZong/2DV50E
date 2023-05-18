from dataset import DRDataset, normalize, load_data


def main():
    train_ds = DRDataset(
        images_folder="../Data/sample_resized_150/",
        path_to_csv="../Data/sample.csv",
        transform=config.val_transforms
    )
    val_ds = DRDataset(
        images_folder="../Data/sample_resized_150/",
        path_to_csv="../Data/sample.csv",
        transform=config.val_transforms
    )
    test_ds = DRDataset(
        images_folder="../Data/sample_resized_150/",
        path_to_csv="../Data/sample.csv",
        transform=config.val_transforms,
        train=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, num_workers=2, shuffle=False
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, shuffle=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, num_workers=2, pin_memory=config.PIN_MEMORY, shuffle=False
    )
    loss_fn = nn.MSELoss()
    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Linear(1536, 5)
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE), model, optimizer, config.LEARNING_RATE)

    model._fc = nn.Linear(1536, 1)

    # Run after training is done and you've achieved good result
    # on validation set, then run train_blend.py file to use information about both eyes concatenated
    # import sys
    # sys.exit()

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        # get on validation
        preds, labels = check_accuracy(val_loader, model, config.DEVICE)
        print(f"QuadraticWeightedKappa (Validation): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        # get on train
        preds, labels = check_accuracy(train_loader, model, config.DEVICE)
        print(f"QuadraticWeightedKappa (Training): {cohen_kappa_score(labels, preds, weights='quadratic')}")

        if config.SAVE_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=config.CHECKPOINT_FILE)

    make_prediction(model, test_loader)

if __name__ == "__main__":
    main()