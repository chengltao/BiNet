import torch
import numpy as np
from torch.utils.data import DataLoader
from load_data import MyDatasets


#function for testing
def test(args, model, device, test_dir):
    print("loading")
    model.eval()
    # loading testing data
    validation_loader = DataLoader(MyDatasets(test_dir), batch_size=args['test_batch_size'], shuffle=True, num_workers=5 )
    lenn = len(validation_loader)
    predicted = []
    target = []
    count = 1
    print("loading finished")
    with torch.no_grad():
        acc = 0
        all_samples = 0
        for item in validation_loader:
            img, tag = item
            outputs = model(img.float().to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            predicted.extend(predict_y.cpu().numpy().tolist())
            tag = tag.cpu()
            all_samples = all_samples + len(tag)
            target.extend(tag.numpy().tolist())
            acc += torch.eq(predict_y, tag.to(device)).sum().item()
            count = count + 1

        a1 = np.array(target)
        b1 = np.array(predicted)
        accuracy1 = sum(a1 == b1) / len(target)
        print("accuracy----------", accuracy1)
    return target, predicted, accuracy1
