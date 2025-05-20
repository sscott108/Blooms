# %%
from dataset import *
from evaluate import *
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# %%
class Weak_Supervision_Dataset():    
    def __init__(self, img_dir,lbl_dir, lat, lon, transforms=None):
        
        self.transforms = transforms
        markers = set()
        self.imgpaths = []
        self.label_paths = {}
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.lat = lat,
        self.lon = lon

        for file in sorted(listdir(self.img_dir)):
            parts = file.split('_')[0]  
            if parts not in markers:
                self.imgpaths.append(os.path.join(self.img_dir, file))
                markers.add(parts)

        for folder in sorted(os.listdir(self.lbl_dir)):
            if folder not in [".DS_Store", ".ipynb_checkpoints"]:
                folder_path = os.path.join(self.lbl_dir, folder)
                for file in sorted(os.listdir(folder_path)):
                    if file.endswith('.tif'):
                        parts = file.split('.')
                        name = parts[-3] 
                        marker = parts[1][:4] + parts[2]  
                        if marker in markers and name == 'CIcyano':
                            if marker not in self.label_paths: 
                                self.label_paths[marker] = os.path.join(folder_path, file)

        self.matches = {}
        for img_path in self.imgpaths:
            img_marker = img_path.split('/')[-1].split('_')[0]  
            if img_marker in markers:
                self.matches[img_marker] = {
                    'image': img_path,
                    'label': self.label_paths.get(img_marker, None)
                }
        
        self.matches = {marker: paths for marker, paths in self.matches.items() if paths['label'] is not None}
        self.keys = list(self.matches.keys())

    def __len__(self): 
        
        return len(self.matches)

    
    def __getitem__(self, idx):
        marker = self.keys[idx]
        
        #image handling 
        with rio.open(self.matches[marker]['image']) as rds:
            data = rds.read([1,2,3])
#             data = np.moveaxis(data, 0, 2) 
            imgdata = np.array(data) #shape (3,600,600)

        #label handling
        with rio.open(self.matches[marker]['label']) as rds:
            data = rds.read()
            crop_size = 1

            transformer = Transformer.from_crs("EPSG:4326", rds.crs, always_xy=True)
            xx, yy = transformer.transform(self.lon, self.lat)
            row, col = rds.index(xx, yy)

            # Define the crop window
            row_start = max(row - crop_size, 0)
            row_end = min(row + crop_size, rds.height)
            col_start = max(col - crop_size, 0)
            col_end = min(col + crop_size, rds.width)

            try:
                window = Window.from_slices((row_start, row_end), (col_start, col_end))
                data = rds.read(window=window)
                data = np.moveaxis(data, 0, 2)  
                lbl_arr = np.array(data)

                fptdata = np.array(int(np.ceil(np.mean(lbl_arr))))
#                 print(fptdata)
#                 fptdata = np.squeeze(lbl_arr)  # Remove extra dimensions of size 1
#                 if fptdata < 31:
#                     class_lbl =1
#                 elif 31 <= fptdata <= 249:
#                     class_lbl = 2
#                 else: class_lbl= 0
                
                if 31 <= fptdata <= 249:
                    class_lbl = 1
                else: class_lbl = 0
                
            except Exception as e:
                print(f"Error processing {self.matches[marker]} label: {e}")
                pass

        sample = {'img': imgdata,
                  'fpt': fptdata,
                  'lbl': class_lbl,
                  'imgfile': self.imgpaths[idx]}

        if self.transforms:
            sample = self.transforms(sample)

        return sample
        
    def display(self, idx):
        #image is tensor now, so use permute
        sample = self[idx]
        
        imgdata = sample['img'].permute(1, 2, 0)
        fptdata = sample['fpt']
        fig,ax = plt.subplots(1,1, figsize=(6,3))
        ax.imshow(imgdata)
        ax.axis('off')
        ax.set_title(f'Sentinel Label: {fptdata.item()}')
        plt.show()
        return fig


def create_dataset(*args, apply_transforms=True, **kwargs):
    if apply_transforms:
        data_transforms = transforms.Compose([
            FlipsAndTricks(),
            CropAndChop(),
#             Normalize(),
            ToTensor(),])
    else: data_transforms = None

    data = Weak_Supervision_Dataset(*args, **kwargs, transforms=data_transforms)
    return data

# %%
lbl_dir = '/datacommons/carlsonlab/srs108/blooms/all_data'

img_dir = '/datacommons/carlsonlab/srs108/crops/arrowhead_beach/'
lat,lon = 36.236102, -76.695387
a=create_dataset(img_dir, lbl_dir, lat, lon, apply_transforms=True)

img_dir = '/datacommons/carlsonlab/srs108/crops/cannons_ferry/'
lat,lon = 36.271557, -76.672967
b=create_dataset(img_dir, lbl_dir, lat, lon, apply_transforms=True)

img_dir = '/datacommons/carlsonlab/srs108/crops/chowan_river/'
lat,lon = 36.055233, -76.686425
c=create_dataset(img_dir, lbl_dir, lat, lon, apply_transforms=True)

img_dir = '/datacommons/carlsonlab/srs108/crops/mt_gould/'
lat,lon = 36.123866, -76.742630
d=create_dataset(img_dir, lbl_dir, lat, lon, apply_transforms=True)

img_dir = '/datacommons/carlsonlab/srs108/crops/point_comfort/'
lat,lon = 36.169458, -76.751827
e=create_dataset(img_dir, lbl_dir, lat, lon, apply_transforms=True)

img_dir = '/datacommons/carlsonlab/srs108/crops/rocky_hock_lane/'
lat,lon = 36.183649, -76.719210
f=create_dataset(img_dir, lbl_dir, lat, lon, apply_transforms=True)

all_chowan = torch.utils.data.ConcatDataset([a,b,c,d,e,f])
val_ratio, test_ratio = 0.25, 0.25
dataset_size = len(all_chowan)
indices = list(range(dataset_size))
batch_size = 16

train_indices, temp_indices = train_test_split(indices, test_size=(val_ratio + test_ratio), shuffle=True)
val_indices, test_indices = train_test_split(temp_indices, test_size=test_ratio/(val_ratio + test_ratio))
train_dataset = Subset(all_chowan, train_indices)
val_dataset = Subset(all_chowan, val_indices)
test_dataset = Subset(all_chowan, test_indices)

# %%
# all_labels = [train_dataset[i]['lbl'] for i in (range(len(train_dataset)))]
# class_counts = np.bincount(all_labels)
# class_weights = 1.0 / class_counts 
# sample_weights = [class_weights[label] for label in all_labels]
# sample_weights = torch.tensor(sample_weights, dtype=torch.float)
# print(class_counts)

# %%
# torch.save((class_counts, class_weights, sample_weights), '2class_weights.pth')
class_counts, class_weights, sample_weights = torch.load('2class_weights.pth')

# %%
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
def train(model, pool, dataloader, criterion, opt, device):
    running_acc = 0
    running_loss = 0
    model.train()

    for i, batch in tqdm(enumerate(dataloader)):
        x = batch['img'].float().to(device)
        y = batch['lbl'].float().to(device) #add dim 1
        output = model(x)

        reduced_output = pool(output)
        reduced_output = torch.flatten(reduced_output, start_dim = 0)

        #loss BCE applies sigmoid to output logits, don't add extra activation.
        loss = criterion(reduced_output,y)
        running_loss += loss.item()

        #accuracy
        output_binary = np.zeros(reduced_output.shape)
        output_binary[reduced_output.cpu().detach().numpy() >= 0] = 1 #if logit is greater than 0, classify as 1, or bloom (THIS BEFORE SIGMOID)
        label = y.cpu().detach().numpy()
        acc = accuracy_score(label,output_binary)
        running_acc += acc
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.empty_cache()
    return running_loss/len(dataloader), running_acc/len(dataloader)

# %%
def test(model, pool, dataloader, criterion, device):
    running_acc = 0
    running_loss = 0
    model.eval()
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            x = batch['img'].float().to(device)
            y = batch['lbl'].float().to(device) #add dim 1
            output = model(x)

            reduced_output = pool(output)
            reduced_output = torch.flatten(reduced_output, start_dim = 0)

            #loss BCE applies sigmoid to output logits, don't add extra activation.
            loss = criterion(reduced_output,y)
            running_loss += loss.item()

            #accuracy
            output_binary = np.zeros(reduced_output.shape)
            output_binary[reduced_output.cpu().detach().numpy() >= 0] = 1 #if logit is greater than 0, classify as 1, or bloom (THIS BEFORE SIGMOID)
            label = y.cpu().detach().numpy()
            acc = accuracy_score(label,output_binary)
            running_acc += acc
            torch.cuda.empty_cache()
    return running_loss/len(dataloader), running_acc/len(dataloader)

# %%
def main(model, epochs, save = False):
    opt                     = optim.SGD(model.parameters(), lr=1e-4, momentum = 0.9)
    criterion               = nn.BCEWithLogitsLoss().to(device)  
    gap                     = nn.AdaptiveAvgPool2d((1, 1))


    best_acc               = 0
    patience                   = np.ceil(0.1*epochs)
    trigger_times              = 0
    history = {'epoch':[],'train_loss': [], 'val_loss':[], 'train_acc': [], 'val_acc':[]}

    for epoch in range(1, epochs+1):

        train_loss, train_acc = train(model, gap, train_loader, criterion, opt, device)
        val_loss, val_acc = test(model, gap, val_loader, criterion, device)
        
        if val_acc > best_acc:
            trigger_times = 0
            best_acc = val_acc
            best_model_weights = model.state_dict()
            if save:
                torch.save(model.state_dict(), 'weak_binary1.pt')

        else:
            trigger_times += 1
            if trigger_times == patience:
                print(f"Early stopping on epoch {epoch} - patience reached")
                break
        
        print(f'Epoch {epoch}\n\tCurrent train/val Loss: {round(train_loss, 5)}/{round(val_loss, 5)}\n\tCurrent train/val Accuracy: {round(train_acc,5)}/{round(val_acc,5)}')
        
        if trigger_times != 0:
            print(f'\tCurrent Triggers: {trigger_times}')

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        df = pd.DataFrame(history)
        df.to_csv('weak.csv', index=False)
# %%
model  = smp.Unet(encoder_name = 'resnet18', encoder_weights = 'imagenet', in_channels = 3,classes = 1).to(device)
# model.load_state_dict(torch.load('weak_binary1.pt'))
main(epochs = 400, model=model, save=True)

# %%
df = pd.read_csv('weak.csv')
avg_t_acc = np.mean(df['train_acc'])
avg_v_acc = np.mean(df['val_acc'])
avg_t_los = np.mean(df['train_loss'])
avg_v_los = np.mean(df['val_loss'])

# %%


# %%
print(f'avg train acc: {avg_tr_acc}\navg val acc: {avg_v_acc}\navg train loss: {avg_t_los}\navg val loss: {avg_v_los}')

# %%
train_val_loss(df['train_loss'], df['val_loss'], len(df['epoch']), save=False)

# %%
train_val_acc(df['train_acc'], df['val_acc'], len(df['epoch']), save=False)

# %%
"""
# Testing
"""

# %%
model  = smp.Unet(encoder_name = 'resnet18', encoder_weights = 'imagenet', in_channels = 3,classes = 1).to(device)
model.load_state_dict(torch.load('weak_binary1.pt'))

# %%
criterion = nn.BCEWithLogitsLoss().to(device) 
pool      = nn.AdaptiveAvgPool2d((1, 1))

all_labels, all_outputs = [],[]
running_acc = 0
running_loss = 0
model.eval()
    
with torch.no_grad():
    for i, batch in tqdm(enumerate(test_loader)):
        x = batch['img'].float().to(device)
        y = batch['lbl'].float().to(device)
        output = model(x)

        output = output.cpu()
        
        fig, ((ax1,ax2), (ax3,ax4), (ax5,ax6)) = plt.subplots(3,2,figsize = (12,12))
        ax1.imshow(output[0].T)
        ax1.set_title(batch['lbl'][0].item())
        ax2.imshow(batch['img'][0].permute(1,2,0))
        
        ax3.imshow(output[1].T)
        ax3.set_title(batch['lbl'][1].item())
        ax4.imshow(batch['img'][1].permute(1,2,0))
        
        ax5.imshow(output[2].T)
        ax5.set_title(batch['lbl'][2].item())
        ax6.imshow(batch['img'][2].permute(1,2,0))
        
        ax1.axis('off')
        ax2.axis('off')
        ax3.axis('off')
        ax4.axis('off')
        ax5.axis('off')
        ax6.axis('off')
        plt.show()
#         reduced_output = pool(output)
#         reduced_output = torch.flatten(reduced_output, start_dim = 0)

#         #loss BCE applies sigmoid to output logits, don't add extra activation.
#         loss = criterion(reduced_output,y)
#         running_loss += loss.item()

#         #accuracy
#         output_binary = np.zeros(reduced_output.shape)
#         output_binary[reduced_output.cpu().detach().numpy() >= 0] = 1 #if logit is greater than 0, classify as 1, or bloom (THIS BEFORE SIGMOID)
#         label = y.cpu().detach().numpy()
#         acc = accuracy_score(label,output_binary)
#         running_acc += acc
#         torch.cuda.empty_cache()
        
#         all_labels.append(label)
#         all_outputs.append(output_binary)  # raw model output

# # Flatten the collected labels and outputs
# all_labels = np.concatenate(all_labels)
# all_outputs = np.concatenate(all_outputs)

# # Calculate AUC
# auc = roc_auc_score(all_labels, all_outputs)
# fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)


# %%
print(auc)

print('Testing Accuracy',running_acc/len(test_loader))
print('Testing Loss',running_loss/len(test_loader))

plot_roc_curve(fpr, tpr, auc, save=False)

# %%
model  = smp.Unet(encoder_name = 'resnet18', encoder_weights = 'imagenet', in_channels = 3,classes = 1)

opt                     = optim.SGD(model.parameters(), lr=1e-4, momentum = 0.9)
criterion               = nn.BCEWithLogitsLoss().to(device)  
gap                     = nn.AdaptiveAvgPool2d((1, 1))

# def train(model, pool, dataloader, criterion, opt, epoch, device):
running_acc = 0
running_loss = 0
model.train()

for i, batch in tqdm(enumerate(train_loader)):
    x = batch['img']
#     y = batch['lbl'].float()
    output = model(x)

    plt.imshow(x[0].T)
    plt.show()
#     reduced_output = gap(output)
#     reduced_output = torch.flatten(reduced_output, start_dim = 0)

#     #loss BCE applies sigmoid to output logits, don't add extra activation.
#     loss = criterion(reduced_output,y)
#     running_loss += loss.item()

#     #accuracy
#     output_binary = np.zeros(reduced_output.shape)
#     output_binary[reduced_output.cpu().detach().numpy() >= 0] = 1 #if logit is greater than 0, classify as 1, or bloom (THIS BEFORE SIGMOID)
#     label = y.cpu().detach().numpy()
#     acc = accuracy_score(label,output_binary)
#     running_acc += acc
#     print(output_binary)
#     print(y)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     torch.cuda.empty_cache()

# %%
