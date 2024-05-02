
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
  # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
             f=model_save_path)

    
    
def save_model_1(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
  # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(model,
             f=model_save_path)
        


def plot_loss_curves(results_bunch):
#def plot_loss_curves(results_bunch: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
   # Setup a plot 
    plt.figure(figsize=(10, 5))
    for i in range(len(results_bunch)):
        results=results_bunch[i]
                   
        # Get the loss values of the results dictionary (training and test)
        loss = results['train_loss']
        test_loss = results['test_loss']

        # Get the accuracy values of the results dictionary (training and test)


        # Figure out how many epochs there were
        epochs = range(len(results['train_loss']))

     
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='con '+str(i))
        if i==len(results_bunch)-1:
            plt.title('Train_Loss')
            plt.xlabel('Epochs')
            plt.legend()

        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, test_loss, label='con '+str(i))
        if i==len(results_bunch)-1:
            plt.title('Test_Loss')
            plt.xlabel('Epochs')
            plt.legend()
    
    plt.show()
    

  


class Data:
    def __init__(self, X, y,sequence_length=1):
        self.X = X
        self.y = y
        self.sequence_length = sequence_length
        
    def __getitem__(self, i): 
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]
    
    def __len__(self):
        return len(self.X)
    
    

def plot_prediction(Pred_Values,True_Values,Lim_value, nbins=15,save_flag=False, save_name=None):
#def plot_loss_curves(results_bunch: dict[str, list[float]]):
    """Plots Results

    Args: True value, Prediction results ,nbins and save flag and save name
        
    """
    xlim =Lim_value
    ylim =Lim_value
    
    
    fig,host=plt.subplots(nrows=2, ncols=3,figsize=(15,8))
    ax0 = host[0][0].twinx()
    
    host[0][0].set_ylim(-7, 8)
    ax0.set_ylim(-0, 10)
    
    host[0][0].set_ylabel("Force[N]")
    ax0.set_ylabel("RMSE[N]")

    
    host[0][0].plot(utils.extraction(True_Values,0),label='True',color='black')
    host[0][0].plot(utils.extraction(Pred_Values,0),label='Predict',color='red',linestyle ="--")
    
    A0=np.linspace(0,len(np.array(utils.extraction(True_Values,0))),len(np.array(utils.extraction(True_Values,0))))
    
    ax0.fill_between(A0,0, np.absolute(np.array(utils.extraction(True_Values,0)) - np.array(utils.extraction(Pred_Values,0))),  label='RMSE',alpha=.3)
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = host[0][0].get_legend_handles_labels()
    lines2, labels2 = ax0.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=0)
    plt.title('X')
    
    
    ax0 = host[0][1].twinx()
    
    host[0][1].set_ylim(-7, 8)
    ax0.set_ylim(-0, 10)
    
    host[0][1].set_ylabel("Force[N]")
    ax0.set_ylabel("RMSE[N]")

    
    host[0][1].plot(utils.extraction(True_Values,1),label='True',color='black')
    host[0][1].plot(utils.extraction(Pred_Values,1),label='Predict',color='blue',linestyle ="--")
    
    A0=np.linspace(0,len(np.array(utils.extraction(True_Values,0))),len(np.array(utils.extraction(True_Values,0))))
    
    ax0.fill_between(A0,0, np.absolute(np.array(utils.extraction(True_Values,1)) - np.array(utils.extraction(Pred_Values,1))),  label='RMSE',alpha=.3)
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = host[0][1].get_legend_handles_labels()
    lines2, labels2 = ax0.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=0)
    plt.title('Y')
    
    ax0 = host[0][2].twinx()
    
    host[0][2].set_ylim(-7, 8)
    ax0.set_ylim(-0, 10)
    
    host[0][2].set_ylabel("Force[N]")
    ax0.set_ylabel("RMSE[N]")

    
    host[0][2].plot(utils.extraction(True_Values,2),label='True',color='black')
    host[0][2].plot(utils.extraction(Pred_Values,2),label='Predict',color='orange',linestyle ="--")
    
    A0=np.linspace(0,len(np.array(utils.extraction(True_Values,2))),len(np.array(utils.extraction(True_Values,2))))
    
    ax0.fill_between(A0,0, np.absolute(np.array(utils.extraction(True_Values,2)) - np.array(utils.extraction(Pred_Values,2))),  label='RMSE',alpha=.3)
    
    # ask matplotlib for the plotted objects and their labels
    lines, labels = host[0][2].get_legend_handles_labels()
    lines2, labels2 = ax0.get_legend_handles_labels()
    ax0.legend(lines + lines2, labels + labels2, loc=0)
    plt.title('Z')
   

    
    #plt.tight_layout(pad=3)
    

    
    #plt.show()
    # For 


    #plt.figure(figsize=(12,3))
    
    y = utils.extraction(Pred_Values,0)
    x = utils.extraction(True_Values,0)
    host[1][0].hist2d(x, y, bins=(nbins, nbins), cmap=plt.cm.jet, range=[[-xlim, xlim], [-ylim, ylim]])
    host[1][0].set_xlabel('Ground Truth Force Magnitude (N)')
    host[1][0].set_ylabel('Predicted Force Magnitude (N)')
    plt.title('X')

    
    y = utils.extraction(Pred_Values,1)
    x = utils.extraction(True_Values,1)
    host[1][1].hist2d(x, y, bins=(nbins, nbins), cmap=plt.cm.jet, range=[[-xlim, xlim], [-ylim, ylim]])
    host[1][1].set_xlabel('Ground Truth Force Magnitude (N)')
    host[1][1].set_ylabel('Predicted Force Magnitude (N)')
    plt.title('Y')


    
    y = utils.extraction(Pred_Values,2)
    x = utils.extraction(True_Values,2)
    host[1][2].hist2d(x, y, bins=(nbins, nbins), cmap=plt.cm.jet,range=[[-xlim, xlim], [-ylim, ylim]])
    host[1][2].set_xlabel('Ground Truth Force Magnitude (N)')
    host[1][2].set_ylabel('Predicted Force Magnitude (N)')
    plt.title('Z')

   
    fig.suptitle(save_name)
    plt.tight_layout(pad=1)
    plt.show()
    if save_flag==True:
        fig.savefig(save_name)
    
'''
def plot_prediction(Pred_Values,True_Values,Lim_value):
#def plot_loss_curves(results_bunch: dict[str, list[float]]):
    """Plots Results

    Args: True value, Prediction results 
        
    """
    xlim =Lim_value
    ylim =Lim_value
    
    

    plt.figure(figsize=(12,6))
    fig, host = plt.subplots(2,3,1)
    ax0 = host.twinx()
    
    host.set_ylim(-7, 8)
    ax0.set_ylim(-0, 10)
    
    host.set_ylabel("Force[N]")
    ax0.set_ylabel("RMSE[N]")

    host.plot(utils.extraction(Pred_Values,0),label='Predict',color='red',linestyle ="--")
    host.plot(utils.extraction(True_Values,0),label='True',color='black')
    
    A0=np.linspace(0,len(np.array(utils.extraction(True_Values,0))),len(np.array(utils.extraction(True_Values,0))))
    
    ax0.fill_between(A0,0, np.absolute(np.array(utils.extraction(True_Values,0)) - np.array(utils.extraction(Pred_Values,0))),  alpha=.3)
    plt.legend(fontsize="12")
    plt.title('X')
    
    
    
    
    fig, host = plt.subplots(2,3,2)
    ax1 = host.twinx()
    
    host.set_ylim(-7, 8)
    ax1.set_ylim(-0, 10)
    
    host.set_ylabel("Force[N]")
    ax1.set_ylabel("RMSE[N]")

    host.plot(utils.extraction(Pred_Values,1),label='Predict',color='blue',linestyle ="--")
    host.plot(utils.extraction(True_Values,1),label='True',color='black')
    
    A1=np.linspace(0,len(np.array(utils.extraction(True_Values,1))),len(np.array(utils.extraction(True_Values,1))))
    
    ax1.fill_between(A1,0, np.absolute(np.array(utils.extraction(True_Values,1)) - np.array(utils.extraction(Pred_Values,1))),  alpha=.3)
    plt.legend(fontsize="12")
    plt.title('Y')
    
    
    
    
    fig, host = plt.subplots(2,3,3)
    ax2 = host.twinx()
    
    host.set_ylim(-7, 8)
    ax2.set_ylim(-0, 10)
    
    host.set_ylabel("Force[N]")
    ax2.set_ylabel("RMSE[N]")

    host.plot(utils.extraction(Pred_Values,2),label='Predict',color='orange',linestyle ="--")
    host.plot(utils.extraction(True_Values,2),label='True',color='black')
    
    A2=np.linspace(0,len(np.array(utils.extraction(True_Values,2))),len(np.array(utils.extraction(True_Values,2))))
    
    ax2.fill_between(A2,0, np.absolute(np.array(utils.extraction(True_Values,2)) - np.array(utils.extraction(Pred_Values,2))),  alpha=.3)
    plt.legend(fontsize="12")
    plt.title('Y')

    # For 



    plt.subplot(2,3,4)
    y = extraction(Pred_Values,0)
    x = extraction(True_Values,0)
    plt.hist2d(x, y, bins=(15, 15), cmap=plt.cm.jet, range=[[-xlim, xlim], [-ylim, ylim]])
    plt.xlabel('Ground Truth Force Magnitude (N)')
    plt.ylabel('Predicted Force Magnitude (N)')
    plt.title('X')

    plt.subplot(2,3,5)
    y = extraction(Pred_Values,1)
    x = extraction(True_Values,1)
    plt.hist2d(x, y, bins=(15, 15), cmap=plt.cm.jet, range=[[-xlim, xlim], [-ylim, ylim]])
    plt.xlabel('Ground Truth Force Magnitude (N)')
    plt.ylabel('Predicted Force Magnitude (N)')
    plt.title('Y')


    plt.subplot(2,3,6)
    y = extraction(Pred_Values,2)
    x = extraction(True_Values,2)
    plt.hist2d(x, y, bins=(15, 15), cmap=plt.cm.jet,range=[[-xlim, xlim], [-ylim, ylim]])
    plt.xlabel('Ground Truth Force Magnitude (N)')
    plt.ylabel('Predicted Force Magnitude (N)')
    plt.title('Z')

    plt.tight_layout(pad=2)

    plt.show()
'''

    
def predict(model: torch.nn.Module, 
            predict_data_loader: torch.utils.data.DataLoader,
            device):
    output = torch.tensor([]).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for X, _ in predict_data_loader:
            y_star = model(X.to(device))
            output = torch.cat((output, y_star), 0)
    return output

def extraction(data,order):
    value=[]
    for i in range(len(data)):
        value.append(data[i][order])
    return value


def predict_data_feature(feature,dataset,Model_address):
    Compare_input_feature=feature
    dataset=dataset
    Model_path_name=Model_address
    
    Y=np.array(dataset.filter(items=For_col))
    X=np.array(dataset.filter(items=Compare_input_feature)) ## important part


    X_scaler = sklearn.preprocessing.MinMaxScaler()
    Y_scaler = sklearn.preprocessing.MinMaxScaler()

    X=torch.FloatTensor(X_scaler.fit_transform(X))
    Y=torch.FloatTensor(Y_scaler.fit_transform(Y))



    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=0.3 # 20% test, 80% train
                                                        #,shuffle=False#,random_state=42
                                                       ) # make the random split reproducible
    train_dataset=DataLoader(utils.Data(X_train,Y_train),batch_size=BATCH_SIZE)
    test_dataset=DataLoader(utils.Data(X_test,Y_test),batch_size=BATCH_SIZE)
    Input_dim=len(X_train[0])
    Output_dim=len(Y_train[0])

    ## Build
    model = model_builder.LSTMModel(
    input_dim = Input_dim,
    hidden_dim=HIDDEN_UNITS,
    layer_dim=2,
    output_dim=Output_dim,
    dropout_prob=0.7)


    model.load_state_dict(torch.load(Model_path_name))
    Pred_Values_right = Y_scaler.inverse_transform(utils.predict(model,DataLoader(utils.Data(X,Y),batch_size=BATCH_SIZE)))

    return Pred_Values_right


def plot_loss_curves(results_bunch):
#def plot_loss_curves(results_bunch: dict[str, list[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
   # Setup a plot 
    plt.figure(figsize=(10, 5))
    for i in range(len(results_bunch)):
        results=results_bunch[i]
                   
        # Get the loss values of the results dictionary (training and test)
        loss = results['train_loss']
        test_loss = results['test_loss']

        # Get the accuracy values of the results dictionary (training and test)


        # Figure out how many epochs there were
        epochs = range(len(results['train_loss']))

     
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='con '+str(i))
        if i==len(results_bunch)-1:
            plt.title('Train_Loss')
            plt.xlabel('Epochs')
            plt.legend()

        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, test_loss, label='con '+str(i))
        if i==len(results_bunch)-1:
            plt.title('Test_Loss')
            plt.xlabel('Epochs')
            plt.legend()

  




def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def predict_data_feature(feature,dataset,Model_address):
    Compare_input_feature=feature
    dataset=dataset
    Model_path_name=Model_address

    Y=np.array(dataset.filter(items=For_col))
    X=np.array(dataset.filter(items=Compare_input_feature)) ## important part


    X_scaler = sklearn.preprocessing.MinMaxScaler()
    Y_scaler = sklearn.preprocessing.MinMaxScaler()

    X=torch.FloatTensor(X_scaler.fit_transform(X))
    Y=torch.FloatTensor(Y_scaler.fit_transform(Y))



    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size=0.3 # 20% test, 80% train
                                                        #,shuffle=False#,random_state=42
                                                       ) # make the random split reproducible
    train_dataset=DataLoader(utils.Data(X_train,Y_train),batch_size=BATCH_SIZE)
    test_dataset=DataLoader(utils.Data(X_test,Y_test),batch_size=BATCH_SIZE)
    Input_dim=len(X_train[0])
    Output_dim=len(Y_train[0])

    ## Build
    model = model_builder.LSTMModel(
    input_dim = Input_dim,
    hidden_dim=HIDDEN_UNITS,
    layer_dim=2,
    output_dim=Output_dim,
    dropout_prob=0.7)


    model.load_state_dict(torch.load(Model_path_name))
    Pred_Values_right = Y_scaler.inverse_transform(utils.predict(model,DataLoader(utils.Data(X,Y),batch_size=BATCH_SIZE)))

    return Pred_Values_right
