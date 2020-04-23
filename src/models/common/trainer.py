
class Trainer():
    def __init__(self):
        pass

    def train():
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_iou = 0.0

        cel = CrossEntropyLoss2d()#weight=class_weights)#my_model.Weighted_BCELoss(pos_weight=[0.0062,1])
        dice= SoftDiceLoss2d()
        focal=FocalLoss(gamma=2)
        # Optimizerの第1引数には更新対象のfc層のパラメータのみ指定
        #optimizer = optim.SGD(list(model.module.conv1.parameters())+list(model.module.fc.parameters()), lr=0.01, momentum=0.9)

        #scheduler = lr_scheduler.StepLR(optimizer, step_size=1800, gamma=0.1)
        loss_dict={x:np.array([np.nan]*num_epochs) for x in ['train', "validation"]}
        iou_dict={x:np.array([np.nan]*num_epochs) for x in ['train', "validation"]}
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # 各エポックで訓練+バリデーションを実行
            for phase in ['train', "validation"]:
                if phase == 'train':
                    #scheduler.step()
                    model.train(True)   # training mode
                else:
                    model.train(False)  # evaluate mode

                running_loss = 0.0
                running_iou = 0
                n_time=time.time()
                for step,data in enumerate(tqdm(dataloaders[phase])):#tqdm(dataloaders[phase]):
                    inputs, labels = data
                    cast_start=time.time()
                    if use_gpu:
                        inputs=torch.Tensor(inputs).to(device)#.unsqueeze_(1)
                        labels = torch.Tensor(labels).to(device)
                    else:
                        inputs=Variable(inputs)#.unsqueeze_(1)
                        labels = Variable(labels)#.float()
                    
                    batch_size,n_input_channel,img_height,img_width=tuple(inputs.shape)
                    optimizer.zero_grad()
                    if phase=='train':
                        outputs=model(inputs)
                    else:
                        with torch.no_grad():
                            outputs=model(inputs)
                    label_weight_sum=labels.sum(dim=(0,2,3))
                    label_weight_sum[label_weight_sum==0]=1
                    class_weights=1/label_weight_sum
                    loss = focal(outputs, labels)#cel(outputs, labels.argmax(1))+dice(outputs,labels).log()#weight=class_weights)(outputs, labels.argmax(1))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    running_loss += loss.data#* batch_size
                    running_iou += iou(outputs,labels).cpu().data.numpy()#*batch_size
                    
                # サンプル数で割って平均を求める
                epoch_loss = running_loss / (step+1)#dataset_sizes[phase]#dataset_sizes[phase]
                epoch_iou = running_iou / (step+1)#dataset_sizes[phase]#dataset_sizes[phase]
                
                print('{} Loss: {:.4f} IOU: {:.4f}'.format(phase, epoch_loss, epoch_iou))
                #print('{} Loss: {:.4f} '.format(phase, epoch_loss))

                loss_dict[phase][epoch]=epoch_loss
                iou_dict[phase][epoch]=epoch_iou
                #visdom
                if phase=="validation":
                    self.plot_image(outputs,labels,epoch)



                # deep copy the model
                # 精度が改善したらモデルを保存する
                if epoch_iou > best_iou:
                    #print("save weights...",end="")
                    best_iou = epoch_iou
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), os.path.join(save_path,"weights.pth"))
                    #print("complete")
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val iou: {:.4f}'.format(best_iou))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def plot_image(self,outputs,labels,epoch):
        output_img=np.zeros((3,img_height,img_width))
        label_img=np.zeros((3,img_height,img_width))
        output_argmax=outputs[0].argmax(0)#(height,width)
        for idx,cla in enumerate(image_datasets["train"].category_list):
            if idx==4:
                break
            #for y in range(img_height):
                #for x in range(img_width):
            #print(cla,labels[0,idx].cpu().data.numpy().sum(),np.array(image_datasets["train"].category_list[cla]).reshape((3,1,1)))
            output_img+=((output_argmax==idx).float().cpu().data.numpy().reshape((1,img_height,img_width))*np.array(image_datasets["train"].category_list[cla]).reshape((3,1,1)))
            # if cla=="car":
            #     print(label_img)
            #     print(labels[0,idx].sum().cpu().data.numpy())
            #     print(image_datasets["train"].category_list[cla])
            label_img+=(labels[0,idx].cpu().data.numpy().reshape((1,img_height,img_width))*np.array(image_datasets["train"].category_list[cla]).reshape((3,1,1)))

            #if idx==3:
                #break
        #print(output_img.shape)
        win_output=viz.image(output_img/255,win="output",opts=dict(title='output'))
        win_label=viz.image(label_img/255,win="label",opts=dict(title='label'))
        win_input=viz.image(inputs[0].cpu().data.numpy(),win="input",opts=dict(title='input'))

        if epoch>0:
            viz.line(X=np.arange(epoch+1),Y=loss_dict["train"][:epoch+1],update="replace",win="loss",name="train")
            viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1],update="replace",win="iou",name="train")
            viz.line(X=np.arange(epoch+1),Y=loss_dict["validation"][:epoch+1],update="replace",win="loss",name="validation")
            viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1],update="replace",win="iou",name="validation")

        else:
            win_loss=viz.line(X=np.arange(epoch+1),Y=loss_dict["train"][:epoch+1],win="loss",name="train")
            win_iou=viz.line(X=np.arange(epoch+1),Y=iou_dict["train"][:epoch+1],win="iou",name="train")
            viz.line(X=np.arange(epoch+1),Y=loss_dict["validation"][:epoch+1],win="loss",name="validation")
            viz.line(X=np.arange(epoch+1),Y=iou_dict["validation"][:epoch+1],win="iou",name="validation")

    def save_model(self):