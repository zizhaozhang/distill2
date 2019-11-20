
import numpy as np
import skimage
import skimage.transform
import skimage.io
import os, sys, traceback, pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import deepdish, shutil
import scipy.misc as misc

left  = 0.02  # the left side of the subplots of the figure
right = 0.99    # the right side of the subplots of the figure
bottom = 0.02   # the bottom of the subplots of the figure
top = 0.99      # the top of the subplots of the figure
wspace = 0.02   # the amount of width reserved for blank space between subplots
hspace = 0.02 # the amount of height reserved for white space between subplots


def _generate_attention_seq1(images, attentions, sents, categories, savedir,
                            sigma=5, threshold=0, upsample_mode='gaussian'):
    ## for tandemnet
    # Inputs:
    #   images [batch_size, h, w, 3]
    #   attentions [[batch_size, spat_width+num_feat] ...] for tandemnet
    #   sents [batch_size]


    max_col = len(attentions) + 1
    spat_len = 49
    for ii in range(len(images)):
        try:
            name = 'img_'+str(ii)
            img = images[ii]
            att = [a[ii] for a in attentions]  # sequence attention
            sent = sents[ii]
            img_atts = [a[:spat_len] for a in att]
            text_atts = [a[spat_len:] for a in att]

            spat_width = int(np.sqrt(att[0].size))
            upscale = int(img.shape[0] / spat_width)

            save_img_path = os.path.join(savedir, name)
            # if os.path.isfile(save_img_path+'_att.png'):
            #     continue

            print ('processing {}/{} {}'.format(ii, len(images), name), flush=True)
            # concept attention
            fig, ax = plt.subplots(1, max_col, sharex='col',sharey='row',figsize=(10*(max_col),10))
            # draw image
            ax[0].imshow(img)
            ax[0].axis('off')
            for i in range(1, max_col):
                # draw attention
                img_att = img_atts[i-1]
                ax[i].imshow(img)
                this_alpha = img_att.copy()
                att_weight = "%.3f"%(this_alpha.sum())
                this_alpha = (this_alpha - this_alpha.min()) / (this_alpha.max() - this_alpha.min())
                this_alpha[this_alpha < threshold] = 0

                if upsample_mode == 'gaussian':
                    alpha_img = skimage.transform.pyramid_expand(this_alpha.reshape(spat_width, spat_width), upscale=upscale, sigma=sigma)
                    alpha_img = skimage.transform.resize(alpha_img, [img.shape[0], img.shape[1]])
                elif upsample_mode == 'nearest':
                    alpha_img = this_alpha.reshape(spat_width, spat_width)
                    alpha_img = misc.imresize(alpha_img, [img.shape[0], img.shape[1]], interp='nearest')

                plt.set_cmap(cm.jet)
                ax[i].imshow(alpha_img, alpha=0.6)
                ax[i].axis('off')

            fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, bottom=bottom + 0.1, right=right, top=top)
            fig.text(0, 0.02, categories[ii], backgroundcolor='white', color='black', fontsize=15)
            fig.savefig(save_img_path+'_imgatt.png')
            plt.close(fig)

            ''' draw text attention  '''
            fig, ax = plt.subplots(1, max_col-1, figsize=(8,5))
            for j in range(max_col-1):
                text_att = text_atts[j]
                x = np.array([a for a in range(text_att.shape[0])])
                text_att = (text_att - text_att.min()) / (text_att.max() - text_att.min() + 0.000000001)
                ax[j].plot(x, text_att, linewidth=6)
                ax[j].set_xticks(x)
                ax[j].set_ylim([0, 1.1])
                ax[j].set_xlabel('Query position', fontsize=15)
                ax[j].set_ylabel('Attention weight', fontsize=15)

            fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, bottom=bottom + 0.4, right=right, top=top)
            # fig.tight_layout()
            for p, se in enumerate(sent):
                fig.text(0, p * 0.06, str(p)+': '+se, backgroundcolor='white', color='black', fontsize=15)

            fig.savefig(save_img_path+'_txtatt.png')
            plt.close(fig)

        except Exception as e:
            print ('--> fail to generate attention maps ... ')
            print (e)
            traceback.print_exc(file=sys.stdout)


def _generate_attention_seq2_deprecated(images, attentions, sents, categories, savedir,
                            sigma=5, threshold=0, upsample_mode='gaussian'):
    # for tandemnet2
    # Inputs:
    #   images [batch_size, h, w, 3]
    #   attentions [[batch_size, sent_num, spat_width] ... ] sent_num = 1 or len(sents[0]) len(attentions) = 1 or num_rn_module
    #   sents [batch_size]

    max_col = attentions[0].shape[1]
    max_row = len(attentions)

    for ii in range(len(images)):
        try:
            name = 'img_'+str(ii)
            img = images[ii]
            img = misc.imresize(img, 0.5) # save space

            att = [a[ii] for a in attentions] # sequence attention
            sent = sents[ii]

            spat_width = int(np.sqrt(att[0].shape[1]))
            upscale = int(img.shape[0] / spat_width)

            save_img_path = os.path.join(savedir, name)
            # if os.path.isfile(save_img_path+'_att.png'):
            #     continue

            print ('processing {}/{} {}'.format(ii, len(images), name), flush=True)
            # concept attention
            max_len = len(att)
            fig, ax = plt.subplots(max_row, max_col+1, sharex='col',sharey='row', figsize=(10*max_col, 10*max_row+len(sent)))

            ax[0,0].imshow(img)
            ax[0,0].axis('off')
            # draw every attention
            for j in range(max_row):
                ax[j,0].axis('off')
                for i in range(max_col):
                    p = ax[j, i+1]
                    p.imshow(img)
                    this_alpha = att[j][i].copy()

                    this_alpha = (this_alpha - this_alpha.min()) / (this_alpha.max() - this_alpha.min())
                    this_alpha[this_alpha < threshold] = 0
                    if upsample_mode == 'gaussian':
                        alpha_img = skimage.transform.pyramid_expand(this_alpha.reshape(spat_width, spat_width), upscale=upscale, sigma=sigma)
                        alpha_img = skimage.transform.resize(alpha_img, [img.shape[0], img.shape[1]])
                    elif upsample_mode == 'nearest':
                        alpha_img = this_alpha.reshape(spat_width, spat_width)
                        alpha_img = misc.imresize(alpha_img, [img.shape[0], img.shape[1]], interp='nearest')

                    # print alpha_img.shape
                    plt.set_cmap(cm.Greys_r)
                    p.imshow(alpha_img, alpha=0.8)
                    p.axis('off')

            fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, bottom=bottom, right=right, top=top)
            # fig.tight_layout()
            fig.text(0, 0.00, categories[ii], backgroundcolor='white', color='red', fontsize=15)
            # fig.subplots_adjust(wspace=wspace, hspace=hspace, left=left, bottom=bottom + 0.4, right=right, top=top)
            # fig.subplots_adjust(bottom=bottom+1)
            # fig.tight_layout()
            for p, se in enumerate(sent):
                fig.text(0, (p+1) * 0.04, str(p) + ': ' + se, backgroundcolor='white', color='black', fontsize=15)

            fig.savefig(save_img_path+'_att.png')
            plt.close(fig)

        except Exception as e:
            print ('--> fail to generate attention maps ... ')
            print (e)
            traceback.print_exc(file=sys.stdout)

def _generate_attention_seq2(images, attentions, sents, categories, savedir,
                            sigma=5, threshold=0, upsample_mode='gaussian'):
    # for tandemnet2
    # Inputs:
    #   images [batch_size, h, w, 3]
    #   attentions [[batch_size, sent_num, spat_width] ... ] sent_num = 1 or len(sents[0]) len(attentions) = 1 or num_rn_module
    #   sents [batch_size]
    #   savedir: partial image paths that contains epoch suffix

    # different from seq_2_deprecated. The attentions arevered along the the same rn module

    if type(attentions) is tuple:
        attentions, attention_class_inds=attentions

    max_col = len(attentions)
    for ii in range(len(images)):
        try:
            name = 'img_'+str(ii)
            img = images[ii]

            att = [a[ii] for a in attentions] # sequence attention
            sent = sents[ii]
            category = categories[ii]
            att_category = []
            # attention_class_inds[ii] may contain label inds not in valid labels of category['Predictions'] (> 0.5)
            # category['Predictions'] is already sorted by the logit

            for ai in attention_class_inds[ii]:
                att_category.append(category['Predictions'][ai] if ai<len(category['Predictions']) else '<None>')

            spat_width = int(np.sqrt(att[0].shape[-1]))
            upscale = int(img.shape[0] / spat_width)

            save_img_path = savedir + '_' + name
            # if os.path.isfile(save_img_path+'_att.png'):
            #     continue

            print ('[attention visualization] processing {}/{} {}'.format(ii, len(images), name), flush=True)
            # concept attention
            max_len = len(att)
            fig, ax = plt.subplots(1, max_col+1, sharex='col',sharey='row', figsize=(10*max_col, 10))

            ax[0].imshow(img)
            ax[0].axis('off')
            # draw every attention
            for i in range(max_col):
                p = ax[i+1]
                p.imshow(img)
                if att[i].size > (spat_width*spat_width) or len(att[i].shape) > 2:
                    # multiple query are averaged
                    this_alpha = att[i].mean(0)
                else:
                    this_alpha = att[i]
                this_alpha = (this_alpha - this_alpha.min()) / (this_alpha.max() - this_alpha.min())
                this_alpha[this_alpha < threshold] = 0
                if upsample_mode == 'gaussian':
                    alpha_img = skimage.transform.pyramid_expand(this_alpha.reshape(spat_width, spat_width), upscale=upscale, sigma=sigma)
                    alpha_img = skimage.transform.resize(alpha_img, [img.shape[0], img.shape[1]])
                elif upsample_mode == 'nearest':
                    alpha_img = this_alpha.reshape(spat_width, spat_width)
                    alpha_img = misc.imresize(alpha_img, [img.shape[0], img.shape[1]], interp='nearest')

                # print alpha_img.shape
                plt.set_cmap(cm.jet)
                p.imshow(alpha_img, alpha=0.6)
                p.set_title(att_category[i], fontsize=30)
                p.axis('off')

            fig.subplots_adjust(wspace=0, hspace=0, left=left, bottom=bottom+0.3, right=right, top=top-0.1)
            fig.text(0, 0.00, 'Labels: {} | Predictions {}'.format(', '.join(category['Labels']), ', '.join(category['Predictions'])),
                    backgroundcolor='white', color='red', fontsize=15)
            # fig.tight_layout()
            for p, se in enumerate(sent):
                fig.text(0, (p+1) * 0.04, str(p) + ': ' + se, backgroundcolor='white', color='black', fontsize=15)

            fig.savefig(save_img_path+'_att.png')
            plt.close(fig)

        except Exception as e:
            # import pdb; pdb.set_trace()
            print ('--> fail to generate attention maps ... ')
            print (e)
            traceback.print_exc(file=sys.stdout)


def generate_attention_sequence(name, images, attentions, sents, categories, savedir,
                            sigma=15, threshold=0, upsample_mode='gaussian'):


    if 'tandemnet2' in name:
        _generate_attention_seq2(images, attentions, sents, categories, savedir,
                              sigma, threshold, upsample_mode)

    elif 'tandemnet' in name:
        _generate_attention_seq1(images, attentions, sents, categories, savedir,
                                sigma, threshold, upsample_mode)

    elif 'resnet' in name:
        _generate_attention_seq2(images, attentions, sents, categories, savedir,
                              sigma, threshold, upsample_mode)
