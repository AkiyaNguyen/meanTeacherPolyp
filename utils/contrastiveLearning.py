import torch

def dice_coefficient(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    return 2 * intersection / (pred.sum() + target.sum() + eps)

def build_pairs(pred_u, pred_l, pred_u_d, pred_l_d):

    pred_img = torch.cat((pred_l, pred_u))
    pred_depth = torch.cat((pred_l_d, pred_u_d))

    N = len(pred_img)
    preds_positive = []
    preds_negative = []


    for i in range(N):
        preds_positive.append((pred_img[i], pred_depth[i]))


    for i in range(N):
        neg_samples = []


        for j in range(N):
            if j != i:
                neg_samples.append(pred_img[j])
                neg_samples.append(pred_depth[j])


        preds_negative.append(neg_samples)

    return preds_positive, preds_negative


def cont_loss(pred_u, pred_l, pred_u_d, pred_l_d, eps=1e-6):

    preds_positive, preds_negative = build_pairs(pred_u, pred_l, pred_u_d, pred_l_d)

    N = len(preds_positive)
    loss = 0.0

    for pos_pair, neg_pair in zip(preds_positive, preds_negative):

        s_pos = dice_coefficient(pos_pair[0], pos_pair[1])


        neg_dice = sum(torch.exp(dice_coefficient(pos_pair[0], neg)) for neg in neg_pair)


        loss += -torch.log(torch.exp(s_pos) / (torch.exp(s_pos) + neg_dice + eps))

    return loss / N