import torch
import tqdm
import yaml
from lib.utils import metrics, collate_fn
from lib.dataloader_aio import ModelAllDataloader, ModelAllDataset
from lib.model import create_model  # Import your model class

def load_model(model, checkpoint_path, device):
    """
    Load a model from a checkpoint file.
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load weights
    model.to(device)
    print("Model loaded successfully!")
    return model, checkpoint

def testing(test_loader, model, device, final_classes):
    """
    Perform testing and calculate metrics using the test dataset.
    """
    print("Starting testing...")
    
    if "__background__" in final_classes:
        idx = final_classes.index("__background__")
        row = idx
        col = idx
    else:
        row = None
        col = None
        
    model.eval()
    record_metrics = metrics(final_classes, score_thres=0.25, iou_thres=0.50, row=row, col=col )
    pbar = tqdm.tqdm(test_loader, total=len(test_loader), position=0, leave=True)
    
    for i, data in enumerate(pbar):
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            preds = model(images)
        
        record_metrics.update(preds, targets)
    
    print("Testing completed.")
    return record_metrics

if __name__ == "__main__":
    # Define dataset and model parameters
    dataset_dir = "F:/optuna/pascal_voc_2012"
    yml_file_path = "F:/optuna/lib/config.yaml"
    with open(yml_file_path, "r") as yml_file:
        config = yaml.load(yml_file, Loader=yaml.FullLoader)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    all_classes = [
        "__background__", "person", "bird", "cat", "cow", "dog", "horse", "sheep", 
        "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", 
        "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"
    ]

    # Initialize dataset and dataloader
    all_datasets = ModelAllDataset(dataset_dir, all_classes)
    test_data = all_datasets.get_testing_data(all_classes, config)
    test_loader = ModelAllDataloader(
        test_data, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    print(f"Number of testing samples: {len(test_data)}\n")

    # Create and load the model
    num_classes = len(all_classes)
    pretrained = True
    model_name = "mobilenet"
    model = create_model(num_classes, pretrained, model_name)

    # Path to the saved model checkpoint
    model_checkpoint_path = "F:/optuna/mobilenet_train_val_WGT_loss_multiplication_optuna_trial000_best.pth"
    model, checkpoint_metadata = load_model(model, model_checkpoint_path, DEVICE)

    # Test the model
    record_metrics = testing(test_loader, model, DEVICE, all_classes)

    # Compute and display results
    record_metrics.compute()
    print("\nConfusion Matrix and Metrics:")
    record_metrics.print()
    record_metrics.plot(key="confusion_matrix", save_dir="output_directory", names=all_classes)
