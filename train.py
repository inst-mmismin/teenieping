import torch
from model import teeniefier
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def main():
    # 학습 세팅값 설정 
    image_size = 28
    train_path = 'dataset/train'
    val_path = 'dataset/val'
    mean = (0.6151984, 0.51760532, 0.46836003)
    std = (0.26411435, 0.24187316, 0.264022790)
    batch_size = 128
    learning_rate = 0.005
    epoch = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 호출 
    ## 전처리 모듈 
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    ## 데이터셋
    train_dataset = ImageFolder(train_path, transform=transform)
    val_dataset = ImageFolder(val_path, transform=transform)
    num_teenieping = len(train_dataset.classes)

    ## 데이터로더 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # 모델 호출
    ## 객체 생성 
    model = teeniefier(num_teenieping=num_teenieping).to(device)

    # 모듈 선언
    ## Loss 
    loss_fn = CrossEntropyLoss()
    ## Optimizer 
    optim = Adam(params=model.parameters(), lr=learning_rate)
    
    # 학습 루프 시작 
    best_acc = 0 
    for i in range(epoch):
        for j, (ping, name) in enumerate(train_loader): 
            ping = ping.to(device)
            name = name.to(device)
            # forward
            predict = model(ping) 
            loss = loss_fn(predict, name)

            # backward 
            loss.backward()
            # optimize 
            optim.step()
            optim.zero_grad()

            # 중간 중간 평가 진행 
            if j % 20 == 0 : 
                print(f'epoch : {i}/{epoch} | step : {j} | loss : {loss:.2f}')
                
                with torch.no_grad(): 
                    model.eval()
                    correct = 0 
                    total = 0 
                    ## 검증용 데이터 활용 
                    for val_ping, val_name in val_loader:
                        val_ping = val_ping.to(device)
                        val_name = val_name.to(device)

                        val_predict = model(val_ping)
                        value, index = torch.max(val_predict, dim=1)

                        correct += (index == val_name).sum().item()
                        total += val_name.shape[0]
                    
                    model.train()
                    
                    ## metric 산출 
                    acc = correct/total

                    ## 성능이 좋으면 
                    if acc > best_acc: 
                        print(f'성능 향상 : {best_acc*100:.1f}% > {acc*100:.1f}%')
                        best_acc = acc 
                        results = {
                            'model' : model.state_dict(),
                            'classes' : train_dataset.classes
                        }
                        ## 저장
                        torch.save(results, 'best_model.ckpt')


if __name__ == '__main__':
    main()