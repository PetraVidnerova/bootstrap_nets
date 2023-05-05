import torch
import torch.nn as nn
import torch.optim as optim

def fit_multi_model(model, train_dl, val_dl, learning_rate=0.001, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.7) 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    
    model.train()

    for epoch in range(1000):
        loss_ep = torch.zeros(model.size)

        print(f"Epoch {epoch} ... ")
        #        with tqdmtotal=len(train_dl)) as t:
        for batch_idx, (data, targets) in enumerate(train_dl):
            data = [
                d.to(device=device)
                for d in data
            ]
            targets = [
                t.repeat(1, model.body[0].repeat).to(device=device)
                for t in targets
            ]
            ## Forward Pass
            scores = model(data)
            loss = [
                criterion(sc, t)
                for sc, t in zip(scores, targets)
            ]
            optimizer.zero_grad()
            _ = [ l.backward() for l in loss]
            optimizer.step()
            loss_ep += torch.Tensor([ l.item() for l in loss])        
            
        print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        with torch.no_grad():
            sum_loss = torch.zeros(model.size)
            print("Computing validation accuracy ...")
            for batch_idx, (data,targets) in enumerate(val_dl):
                data = [
                    d.to(device=device)
                    for d in data
                ]
                targets = [
                    t.repeat(1, model.body[0].repeat).to(device=device)
                    for t in targets
                ]
                ## Forward Pass
                scores = model(data)
                loss = [
                    criterion(sc, t)
                    for sc, t in zip(scores, targets)
                ]
                sum_loss += torch.Tensor([ l.item() for l in loss])        

            print(
                f"VAL loss: { sum_loss / len(val_dl)}"
            )

    

def fit_models(model, train_dl, val_dl, learning_rate=0.001, batch_size=32):

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr= learning_rate, momentum=0.7) 

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    
    model.train()

    for epoch in range(1000):
        loss_ep = 0

        print(f"Epoch {epoch} ... ")
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
            # if epoch > 0 and epoch % 300 == 0:
            #     for g in optimizer.param_groups:
            #         print(f"**** {g['lr']}")
            #         g['lr'] = 0.9*g['lr']
    
        
            
        print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")

        with torch.no_grad():
            sum_loss = 0
            print("Computing validation accuracy ...")
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
            print(
                f"VAL loss: {float(sum_loss) / len(val_dl)  :.2f}"
            )

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
    print(
        f"TEST accuracy: {sum_loss.min() :.2f}"
    )
    print(sum_loss)
    return sum_loss.min(), sum_loss.argmin()
   
def eval_model(model, test_dl):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()
    
    model.eval()
    sum_loss = 0
    outputs = []
    with torch.no_grad():
        for data, targets in test_dl:
            data = data.to(device=device)
            targets = targets.to(device=device)
            # Forward Pass
            scores = model(data)
            outputs.append(scores)
            # geting predictions
            sum_loss += criterion(scores, targets).item()
    print(
        f"TEST accuracy: {float(sum_loss) / len(test_dl) :.2f}"
    )
    return outputs
