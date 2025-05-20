# %%
from abemarle_utils import *

# %%
x = create_dataset(imgdir = '/datacommons/carlsonlab/srs108/blooms/downloads',
    lbldir= '/datacommons/carlsonlab/srs108/blooms/lbls',
    apply_transforms=True)

# %%
val_ratio, test_ratio =0.15, 0.15
dataset_size = len(x)
indices = list(range(dataset_size))

train_indices, temp_indices = train_test_split(indices, test_size=(val_ratio + test_ratio), shuffle=True)
val_indices, test_indices = train_test_split(temp_indices, test_size=test_ratio/(val_ratio + test_ratio))
train_dataset = Subset(x, train_indices)
val_dataset = Subset(x, val_indices)
test_dataset = Subset(x, test_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
print(len(train_dataset), len(val_dataset), len(test_dataset))

# %%
def train(model, dataloader, criterion, opt, epoch, device):
    running_acc = 0
    running_loss = 0
    train_ious = []
    model.train()
    
    for i, batch in tqdm(enumerate(dataloader)):
        x = batch['img'].float().to(device)
        y = batch['fpt'].float().squeeze(dim=1).to(device) #add dim 1

        output = model(x)
        #loss
        loss = criterion(output,y.long())
        running_loss += loss.item()
        
        #mIoU
        total_iou, iou_list = mIOU(y, output, num_classes=5)
        train_ious.append(total_iou)
        
        #pixel accuracy
        acc_epoch = pixel_accuracy(output, y)
        running_acc += acc_epoch.item()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    return running_loss/len(dataloader), running_acc/len(dataloader), np.average(train_ious)

# %%
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0
    running_acc = 0
    test_ious = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            x = batch['img'].float().to(device)
            y = batch['fpt'].float().squeeze(dim=1).to(device) #add dim 1

            output = model(x)
            
            loss = criterion(output,y.long())
            running_loss += loss.item()
            
            total_iou, _ = mIOU(y, output, num_classes=5)
            test_ious.append(total_iou)
            
            acc_epoch = pixel_accuracy(output, y)
            running_acc += acc_epoch.item()
            
            torch.cuda.empty_cache()
        return running_loss/len(dataloader), running_acc/len(dataloader), np.average(test_ious)
            

# %%
def main(model, epochs):
    lr                         = 1e-4

    opt                        = optim.SGD(model.parameters(), lr=lr, momentum = 0.99)
    criterion                  = nn.CrossEntropyLoss().to(device)  
    scheduler                  = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.5, threshold=lr, min_lr=1e-6)
    best_iou = 0.0 

    patience                   = np.ceil(0.10*epochs)
    trigger_times              = 0
    history = {'epoch':[],'train_loss': [], 'val_loss':[], 
               'train_iou': [], 'val_iou':[], 'train_acc': [], 'val_acc':[]}

    running_acc = 0
    train_loss_values, val_loss_values, train_ious, val_ious = [], [], [], []
    for epoch in range(1, epochs+1):

            train_loss, train_acc, train_iou = train(model, train_loader, criterion, opt, epoch, device)
            val_loss, val_acc, val_iou = test(model, val_loader, criterion, device)


            if val_iou > best_iou:
                trigger_times = 0
                best_iou = val_iou
                best_model_weights = model.state_dict()
                torch.save(model.state_dict(), 'best_model_weights_5class.pt')

            else:
                trigger_times += 1
                print(f"Triggered on epoch {epoch}: {trigger_times}/{patience} with iou {round(val_iou,5)}, current best {round(best_iou, 5)}")
                if trigger_times >= patience:
                    print(f"Early stopping on epoch {epoch} - patience reached")
                    break
                    
            
            if epoch % 25 ==0:
                print(f'Epoch {epoch}\n\tCurrent train/val IoU: {round(train_iou,5)}/{round(val_iou,5)}\n\tCurrent train/val Loss: {round(train_loss, 5)}/ {round(val_loss, 5)}')
                    
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_iou'].append(train_iou)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)
            history['val_acc'].append(val_acc)

    df = pd.DataFrame(history)
    return df

# %%
model  = smp.Unet(encoder_name = 'resnet18', encoder_weights = 'imagenet', in_channels = 3,classes = 5).to(device)


# %%
rundf = main(epochs=500, model=model)
# %time

# %%
rundf.to_csv('history.csv', index=False)

# %%
"""
# Post Run
"""

# %%
# df = pd.read_csv('history.csv')

# %%
# def train_val_iou(loss_train, loss_val, epochs, save = True, fig_name=''):
#     epoch = range(epochs)
#     fig, ax = plt.subplots(1,1, figsize = (10,6))   
#     ax.plot(epoch, loss_train, color='b', linewidth=0.5, label='Training')
#     ax.plot(epoch, loss_val, color='r', linewidth=0.5, label='Validation')

#     ax.set_xlabel('Iters')
#     ax.set_ylabel('Iou')
#     ax.set_title('Training and Validation Intersection-over-Union')
#     ax.legend()
#     plt.show()
#     if save==True:
#         fig.savefig(fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')

# # %%
# train_val_iou(df['train_iou'], df['val_iou'], len(df['epoch']), save=False)

# # %%
# def train_val_loss(loss_train, loss_val, epochs, save = True, fig_name=''):
#     epoch = range(epochs)
#     fig, ax = plt.subplots(1,1, figsize = (10,6))   
#     ax.plot(epoch, loss_train, color='b', linewidth=0.5, label='Training')
#     ax.plot(epoch, loss_val, color='r', linewidth=0.5, label='Validation')

#     ax.set_xlabel('Iters')
#     ax.set_ylabel('Loss')
#     ax.set_title('Training and Validation Loss')
#     ax.legend()
#     plt.show()
#     if save==True:
#         fig.savefig(fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')

# # %%
# train_val_loss(df['train_loss'], df['val_loss'], len(df['epoch']), save=False)

# # %%
# def train_val_acc(loss_train, loss_val, epochs, save = True, fig_name=''):
#     epoch = range(epochs)
#     fig, ax = plt.subplots(1,1, figsize = (10,6))   
#     ax.plot(epoch, loss_train, color='b', linewidth=0.5, label='Training')
#     ax.plot(epoch, loss_val, color='r', linewidth=0.5, label='Validation')

#     ax.set_xlabel('Iters')
#     ax.set_ylabel('Accuracy')
#     ax.set_title('Training and Validation Accuracy')
#     ax.legend()
#     plt.show()
#     if save==True:
#         fig.savefig(fig_name+'.png', transparent=False, facecolor='white', bbox_inches='tight')

# # %%
# train_val_acc(df['train_acc'], df['val_acc'], len(df['epoch']), save=False)

# # %%
# model.load_state_dict(torch.load('weights/best_model_weights.pt'))

# # %%
# criterion = nn.CrossEntropyLoss().to(device)  

# test_loss, test_acc,test_iou = test(model, test_loader, criterion, device)
# print('Testing IoU:',test_iou)
# print('Testing Accuracy',test_acc)


# # %%
# from torchmetrics import JaccardIndex
# iou = JaccardIndex(task='multiclass', num_classes=4).to(device)

# # %%
# model.eval()
# running_loss = 0
# running_acc = 0
# test_ious = []
# with torch.no_grad():
#     for i, batch in tqdm(enumerate(test_loader)):
#         x = batch['img'].float().to(device)
#         y = batch['fpt'].float().squeeze(dim=1).to(device) #add dim 1

#         output = model(x)
        
#         print(iou(output, y))
#         print(output.shape)
                    
#         total_iou, ious = mIOU(y, output, num_classes=4)
#         print(ious)
#         softmax = nn.Softmax(dim=1)
#         preds = torch.argmax(softmax(output),axis=1).to('cpu')
        
#         preds1 = np.array(preds[0,:,:])
        
#         fig, ax = plt.subplots(1,2)
#         ax[0].imshow(preds1)
#         ax[0].set_title('Prediction')
#         ax[0].axis('off')
#         ax[1].imshow(y[0,:,:].to('cpu'))
#         ax[1].set_title('Labeled Truth')
#         ax[1].axis('off')
#         plt.show()
#         torch.cuda.empty_cache()


# # %%
# df = pd.read_csv('history.csv')