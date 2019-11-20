import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('cmd', choices=['train', 'test'], default='train')
parser.add_argument('-d', '--data-dir', default='./dataset')
parser.add_argument('-c', '--classes', default=0, type=int)
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--cnn_lr', type=float, default=0.01, metavar='CNNLR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--mm_lr', type=float, default=0.0001, metavar='MMLR',
                    help='learning rate (default: 0.0001)')

parser.add_argument('-e', '--evaluate', dest='evaluate',
                    action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--name', default='model')
parser.add_argument('--save_result', action='store_true')
parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
parser.add_argument('--no_pretrained', action='store_true')
parser.add_argument('--lr_decay', type=int, default=1)
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--lr_decay_at', type=str, default='')
parser.add_argument('--no_history', action='store_true')

## use for distill models
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--embed_size', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--multifeat_size', type=int, default=512)
parser.add_argument('--attfeat_size', type=int, default=512)


parser.add_argument('--no_mm', action='store_true', help='if not use multimodal while a simple cnn. see main.py')
parser.add_argument('--grad_clip', type=float, default=0.1)
parser.add_argument('--fix_cnn', action='store_true', help='if fix the main cnn network')
parser.add_argument('--loss_mult', type=int, default=1, help='multiply the loss by this amount')

parser.add_argument('--num_rn_module', type=int, default=3, help='relation module in parallel.')
parser.add_argument('--use_text_in_test', action='store_true', help='')
parser.add_argument('--textimg_drop_rate', type=float, default=0.1)
parser.add_argument('--death_rate', type=float, default=0.5)
parser.add_argument('--base_cnn_model', type=str, default='resnet101', help='the pretrained baseline cnn model')
parser.add_argument('--dataset', type=str, default='', help='which dataset [coco, vgnome, bcidr, chestxray]')
parser.add_argument('--multi_drop_rate', type=float, default=0.0)
parser.add_argument('--last_drop_rate', type=float, default=0.0)
parser.add_argument('--dynamic_deathrate', action='store_true')
parser.add_argument('--save_attention', action='store_true', help='save_attention_in_disk')
parser.add_argument('--f1_topk', type=int, default=3, help='top-k results for evaluation')
parser.add_argument('--loader_unary_mode', action='store_true', help='if load image with text one by one')

args = parser.parse_args()

if args.dataset in ['bcidr'] and args.num_rn_module > 1:
    print ('WARNING: args.num_rn_module is suggested to be 1 for single label classification datasets')
if args.num_rn_module == 1 and args.textimg_drop_rate != 0:
    print ('WARNING: args.textimg_drop_rate == 0 if num_rm_module == 1, set textimg_drop_rate = 0')
