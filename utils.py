from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def fit_multi_model(model, train_dl, val_dl, learning_rate=0.001, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.7) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, min_lr=1e-9)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    
    model.train()

    for epoch in range(300):
        loss_ep = 0

        print(f"Epoch {epoch} ... ")
        with tqdm(total=len(train_dl)) as t:
            for batch_idx, (data, targets) in enumerate(train_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = model(data)
                loss = criterion(scores, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_ep += loss.item()       
                t.update()
        print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        with torch.no_grad():
            sum_loss = 0
            print("Computing validation accuracy ...")
            for batch_idx, (data,targets) in enumerate(val_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                ## Forward Pass
                scores = model(data)
                loss = criterion(scores, targets)
                sum_loss += loss.item()

            print(
                f"VAL loss: { sum_loss / len(val_dl)}"
            )
            scheduler.step(sum_loss)
            

def fit_models(model, train_dl, val_dl, learning_rate=0.001, batch_size=32):

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.7) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, min_lr=1e-9)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    
    model.train()

    for epoch in range(300):
        loss_ep = 0

#        print(f"Epoch {epoch} ... ")
        #        with tqdmtotal=len(train_dl)) as t:
        for batch_idx, (data, targets) in enumerate(train_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            multi_targets = targets.repeat(1, model.repeat)
            ## Forward Pass
            scores = model(data)
            loss = criterion(scores, multi_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ep += loss.item()
        
            
 #       print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        with torch.no_grad():
            sum_loss = 0
 #           print("Computing validation accuracy ...")
            #            with tqdm(total=len(val_dl)) as t:
            for batch_idx, (data,targets) in enumerate(val_dl):
                data = data.to(device=device)
                targets = targets.to(device=device)
                multi_targets = targets.repeat(1, model.repeat)
                ## Forward Pass
                scores = model(data)
#                print(scores, targets)
                sum_loss += criterion(scores, multi_targets).item() #((scores-targets)**2).sum()
            #       t.update()
            # print(
            #     f"VAL loss: {float(sum_loss) / len(val_dl)  :.2f}"
            # )
            scheduler.step(sum_loss)

def eval_models(model, test_dl):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    criterion = nn.MSELoss()

    model.eval()
    sum_loss = torch.zeros(model.repeat).to(device)
    with torch.no_grad():
        for data, targets in test_dl:
            data = data.to(device=device)
            targets = targets.to(device=device)
            # Forward Pass
            scores = model.split_eval(data)
            # geting predictions
            sum_loss += torch.cat([
                criterion(single_scores, targets).reshape(1)
                for single_scores in scores
            ])
    sum_loss /= len(test_dl)
    # print(
    #     f"TEST accuracy: {sum_loss.min() :.2f}"
    # )
    # print(sum_loss)
    return sum_loss.min(), sum_loss.argmin()
   
def eval_model(model, test_dl):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    
    model.eval()
    sum_loss = 0
    residuals = []
    with torch.no_grad():
        for data, targets in test_dl:
            data = data.to(device=device)
            targets = targets.to(device=device)
            # Forward Pass
            scores = model(data)
            residuals.append(targets-scores)
            # geting predictions
            sum_loss += criterion(scores, targets).item()
    # print(
    #     f"TEST accuracy: {float(sum_loss) / len(test_dl) :.2f}"
    # )
    return torch.cat(residuals)
