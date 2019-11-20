from data.data_loader import get_loader
import torchvision.transforms as transforms
import torch
from trainer import Trainer
# get args
from opts import args

devices = [a for a in range(torch.cuda.device_count())]
print(args)

# init data loader
train_loader, num_samples, vocab_size = get_loader(args.data_dir, args.dataset, 'train',
                                                    batch_size=args.batch_size*len(devices),
                                                    shuffle=True, num_workers=4)
val_loader, num_samples, vocab_size=get_loader(args.data_dir, args.dataset, 'test',
                                                batch_size=args.batch_size * len(devices),
                                                shuffle=False, num_workers=4 if args.cmd == 'train' else 0,
                                                unary_mode=args.loader_unary_mode)

# init model
if not args.no_mm:
    if 'tandemnet2v2' in args.name:
        from model.tandemnet2_v2 import DistillModel
    elif 'tandemnetv2' in args.name:
        from model.tandemnet_v2 import DistillModel

    model = DistillModel(args, vocab_size, n_classes=train_loader.dataset.num_cats, model_name=args.base_cnn_model).cuda()
else:
    from model.tandemnet import MultiLabelResNet
    model = MultiLabelResNet(args=args, n_classes=train_loader.dataset.num_cats, model_name=args.base_cnn_model).cuda()

trainer = Trainer(model, train_loader, val_loader, args=args, devices=devices)

if args.cmd == 'train':
    trainer.train()
elif args.cmd == 'test':
    trainer.test()
