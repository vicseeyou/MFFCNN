
import torch.cuda
from model import *
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from read_data import *
from metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def jiaolong(net, optim, epoch, loss, dirs, dir_name):
    net = net.to(device)
    loss = loss.to(device)


    acc_is, f1_is, auc_is, cross_is, meanabs_is, meansqu_is = 0, 0, 0, 10, 10, 10

    for i in range(epoch):
        print("进度：{}%".format(round((i + 1) / epoch * 100, 3)))

        train_model(train_data, net, optim, loss)

        test_acc, test_f1, test_auc, test_cross, test_meanabs, test_meansqu = test_model(test_data, net)


        if test_acc > acc_is :
            acc_is = test_acc
            verify_acc, verify_f1, verify_auc, verify_cross, verify_meanabs, verify_meansqu = verify_model(verify_data, net)

            # torch.save(net.state_dict(), rf'{dirs}/model/{dir_name}/{test_acc}+{i}.pth')
            torch.save(net, rf'{dirs}/model/{dir_name}/{verify_acc}+{i}.pth')


def train_model(train_data, net, optim, loss):

    toral_loss = 0
    net.train()

    for data in train_data:
        imgs1, imgs2, targets = data
        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)
        targets = targets.to(device)
        outputs = net(imgs1, imgs2)
        result_loss = loss(outputs, targets)

        optim.zero_grad()
        result_loss.backward()
        optim.step()
        toral_loss +=result_loss.item()

    return toral_loss

def verify_model(verify_data, net):

    net.eval()
    all_preds = []
    all_targets = []
    all_probs = []  # AUC

    with torch.no_grad():
        for data in verify_data:
            imgs1, imgs2, targets = data
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            targets = targets.to(device)
            outputs = net(imgs1, imgs2)

            probs = F.softmax(outputs, dim=1)

            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # accuracy
    acc = (all_preds == all_targets).mean()
    # F1
    f1 = f1_score(all_targets, all_preds, average='macro')
    # CE
    cross_entropy = F.cross_entropy(torch.tensor(all_probs), torch.tensor(all_targets))
    # AUC
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    # MAE
    mae = mean_absolute_error(all_targets, all_preds)
    # MSE
    mse = mean_squared_error(all_targets, all_preds)
    # return
    return acc, f1, cross_entropy.item(), auc, mae, mse


def test_model(test_data, net):

    net.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for data in test_data:
            imgs1, imgs2, targets = data
            imgs1 = imgs1.to(device)
            imgs2 = imgs2.to(device)
            targets = targets.to(device)
            outputs = net(imgs1, imgs2)

            # logits
            probs = F.softmax(outputs, dim=1)  # Softmax

            #
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    #
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # accuracy
    acc = (all_preds == all_targets).mean()
    # F1
    f1 = f1_score(all_targets, all_preds, average='macro')
    # CE
    cross_entropy = F.cross_entropy(torch.tensor(all_probs), torch.tensor(all_targets))
    # AUC
    auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
    # MAE
    mae = mean_absolute_error(all_targets, all_preds)
    # MSE
    mse = mean_squared_error(all_targets, all_preds)
    # return
    return acc, f1, cross_entropy.item(), auc, mae, mse

name = rf"MFFCNN"   #SFCNN MFFCNN

data1_063 = rf"Gabor/15X15/win_25/063"
data1_064 = rf"Gabor/15X15/win_25/064"
data1_065 = rf"Gabor/15X15/win_25/065"
data1_240 = rf"Gabor/15X15/win_25/240"
data1_241 = rf"Gabor/15X15/win_25/241"
data1_242 = rf"Gabor/15X15/win_25/242"

data2_063 = rf"Gabor/15X15/win_75/063"
data2_064 = rf"数Gabor/15X15/win_75/064"
data2_065 = rf"Gabor/15X15/win_75/065"
data2_240 = rf"Gabor/15X15/win_75/240"
data2_241 = rf"Gabor/15X15/win_75/241"
data2_242 = rf"Gabor/15X15/win_75/242"

label_1 = r"WELL1_labels_119_prob.xlsx"
label_2 = r"WELL2_labels_96_prob.xlsx"

data063 = read_multi_scale_img(data1_063, data2_063, label_1)
data064 = read_multi_scale_img(data1_064, data2_064, label_1)
data065 = read_multi_scale_img(data1_065, data2_065, label_1)
data240 = read_multi_scale_img(data1_240, data2_240, label_2)
data241 = read_multi_scale_img(data1_241, data2_241, label_2)
data242 = read_multi_scale_img(data1_242, data2_242, label_2)

verify_data =data063 + data240
train_data = data064 + data241
test_data = data065 + data242

test_data_size = len(test_data)
train_data_size = len(train_data)
verify_data_size = len(verify_data)

batch_size = 64
train_data = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_data = DataLoader(test_data, batch_size=batch_size, shuffle=False)
verify_data = DataLoader(verify_data, batch_size=batch_size, shuffle=False)

net = MFFCNN()

learning_rate = 0.01
optim = torch.optim.SGD(net.parameters(), lr=learning_rate)
#loss
loss = nn.CrossEntropyLoss()
epoch = 6000

dir_name = rf"{learning_rate}_10"
dirs = rf"results/{name}"
log_dir = rf"{dirs}/logs/{dir_name}"
model_dir = rf"{dirs}/model/{dir_name}"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    print(f"file OK!")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"file OK!")


jiaolong(net, optim, epoch, loss, dirs, dir_name)