import os
import numpy as np
import torch
import torch.nn.functional as F

try:
    import gco

    GCO_AVAILABLE = True
except Exception:
    gco = None
    GCO_AVAILABLE = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def to_one_hot(inp, num_classes, device='cuda'):
    '''one-hot label'''
    y_onehot = torch.zeros((inp.size(0), num_classes), dtype=torch.float32, device=device)
    y_onehot.scatter_(1, inp.unsqueeze(1), 1)
    return y_onehot


def cost_matrix(width, device='cuda'):
    '''transport cost'''
    C = np.zeros([width ** 2, width ** 2], dtype=np.float32)

    for m_i in range(width ** 2):
        i1 = m_i // width
        j1 = m_i % width
        for m_j in range(width ** 2):
            i2 = m_j // width
            j2 = m_j % width
            C[m_i, m_j] = abs(i1 - i2) ** 2 + abs(j1 - j2) ** 2

    C = C / (width - 1) ** 2
    C = torch.tensor(C)
    if device == 'cuda':
        C = C.cuda()

    return C


cost_matrix_dict = {'2': cost_matrix(2, device).unsqueeze(0), '4': cost_matrix(4, device).unsqueeze(0),
                    '8': cost_matrix(8, device).unsqueeze(0), '16': cost_matrix(16, device).unsqueeze(0),
                    '32': cost_matrix(32, device).unsqueeze(0), '64': cost_matrix(64, device).unsqueeze(0)}


def mixup_process(out, target_reweighted, hidden=0, args=None, grad=None, noise=None, adv_mask1=0, adv_mask2=0,
                  mp=None):
    '''various mixup process'''
    if args is not None:
        mixup_alpha = args.mixup_alpha
        in_batch = args.in_batch
        mean = args.mean
        std = args.std
        box = args.box
        graph = args.graph
        beta = args.beta
        gamma = args.gamma
        eta = args.eta
        neigh_size = args.neigh_size
        n_labels = args.n_labels
        transport = args.transport
        t_eps = args.t_eps
        t_size = args.t_size
        # [新增] 获取锚定强度参数，默认为15.0
        anchor_lambda = getattr(args, 'anchor_lambda', 15.0)

    block_num = 2 ** np.random.randint(1, 5)
    indices = np.random.permutation(out.size(0))

    lam = get_lambda(mixup_alpha)
    if hidden:
        # Manifold Mixup
        out = out * lam + out[indices] * (1 - lam)
        ratio = torch.ones(out.shape[0], device='cuda') * lam
        # 为hidden模式提供一个默认mask，防止未定义报错
        b, _, h, w = out.shape
        mask = torch.ones((b, 1, h, w), device=out.device, dtype=out.dtype)
        target_reweighted_final = target_reweighted * lam + target_reweighted[indices] * (1 - lam)  # 简单近似
    else:
        target_shuffled_onehot = target_reweighted[indices].clone()
        if box:
            # CutMix
            out, ratio = mixup_box(out, out[indices], alpha=lam, device='cuda')
            b, _, h, w = out.shape
            mask = torch.ones((b, 1, h, w), device=out.device, dtype=out.dtype)
            target_reweighted_final = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
        elif graph:
            # PuzzleMix
            if (block_num > 1):
                # 1. 提取 Scribble (兼容 One-hot 和 Long 格式)
                if target_reweighted.dim() == 4 and target_reweighted.shape[1] > 1:
                    scribble1 = target_reweighted.argmax(1)
                    scribble1[target_reweighted.sum(1) == 0] = 255
                    scribble2 = target_reweighted[indices].argmax(1)
                    scribble2[target_reweighted[indices].sum(1) == 0] = 255
                else:
                    scribble1 = target_reweighted.squeeze(1).clone()
                    scribble2 = target_reweighted[indices].squeeze(1).clone()

                # 2. 核心调用 (确保 out, ratio, mask = ... 这一句完整)
                out, ratio, mask = mixup_graph(out, target_reweighted, grad, indices, block_num=block_num,
                                               alpha=lam, beta=beta, gamma=gamma, eta=eta, neigh_size=neigh_size,
                                               n_labels=n_labels,
                                               mean=mean, std=std, transport=transport, t_eps=t_eps, t_size=t_size,
                                               noise=noise, adv_mask1=adv_mask1, adv_mask2=adv_mask2, mp=mp,
                                               device='cuda',
                                               scribble1=scribble1, scribble2=scribble2, anchor_lambda=anchor_lambda)

                target_reweighted_final = torch.zeros_like(target_shuffled_onehot)
                for i in range(target_shuffled_onehot.shape[1]):
                    target_reweighted_final[:, i, :, :] = target_reweighted[:, i, :, :] * mask[:, 0, :,
                                                                                          :] + target_shuffled_onehot[:,
                                                                                               i, :, :] * (
                                                                      1 - mask[:, 0, :, :])
            else:
                # Fallback (理论上block_num总是>1，但为了安全起见)
                out = out * lam + out[indices] * (1 - lam)
                target_reweighted_final = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
                b, _, h, w = out.shape
                mask = torch.ones((b, 1, h, w), device=out.device, dtype=out.dtype)
        else:
            # Input Mixup
            out = out * lam + out[indices] * (1 - lam)
            target_reweighted_final = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
            b, _, h, w = out.shape
            mask = torch.ones((b, 1, h, w), device=out.device, dtype=out.dtype)

    return out, target_reweighted_final, indices, mask


def get_lambda(alpha=1.0, alpha2=None):
    '''Return lambda'''
    if alpha > 0.:
        if alpha2 is None:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = np.random.beta(alpha + 1e-2, alpha2 + 1e-2)
    else:
        lam = 1.
    return lam


def graphcut_multi(unary1, unary2, pw_x, pw_y, alpha, beta, eta, n_labels=2):
    '''alpha-beta swap algorithm'''
    block_num = unary1.shape[0]

    large_val = 1000 * block_num ** 2

    if n_labels == 2:
        prior = eta * np.array([-np.log(alpha + 1e-8), -np.log(1 - alpha + 1e-8)]) / block_num ** 2
    elif n_labels == 3:
        prior = eta * np.array([-np.log(alpha ** 2 + 1e-8), -np.log(2 * alpha * (1 - alpha) + 1e-8),
                                -np.log((1 - alpha) ** 2 + 1e-8)]) / block_num ** 2
    elif n_labels == 4:
        prior = eta * np.array([-np.log(alpha ** 3 + 1e-8), -np.log(3 * alpha ** 2 * (1 - alpha) + 1e-8),
                                -np.log(3 * alpha * (1 - alpha) ** 2 + 1e-8),
                                -np.log((1 - alpha) ** 3 + 1e-8)]) / block_num ** 2

    unary_cost = (large_val * np.stack(
        [(1 - lam) * unary1 + lam * unary2 + prior[i] for i, lam in enumerate(np.linspace(0, 1, n_labels))],
        axis=-1)).astype(np.int32)
    pairwise_cost = np.zeros(shape=[n_labels, n_labels], dtype=np.float32)

    for i in range(n_labels):
        for j in range(n_labels):
            pairwise_cost[i, j] = (i - j) ** 2 / (n_labels - 1) ** 2

    pw_x = (large_val * (pw_x + beta)).astype(np.int32)
    pw_y = (large_val * (pw_y + beta)).astype(np.int32)

    labels = 1.0 - gco.cut_grid_graph(unary_cost, pairwise_cost, pw_x, pw_y, algorithm='swap') / (n_labels - 1)
    mask = labels.reshape(block_num, block_num)

    return mask


def neigh_penalty(input1, input2, k):
    '''data local smoothness term'''
    pw_x = input1[:, :, :-1, :] - input2[:, :, 1:, :]
    pw_y = input1[:, :, :, :-1] - input2[:, :, :, 1:]

    pw_x = pw_x[:, :, k - 1::k, :]
    pw_y = pw_y[:, :, :, k - 1::k]

    pw_x = F.avg_pool2d(pw_x.abs().mean(1), kernel_size=(1, k))
    pw_y = F.avg_pool2d(pw_y.abs().mean(1), kernel_size=(k, 1))

    return pw_x, pw_y


def mixup_box(input1, input2, alpha=0.5, device='cuda'):
    '''CutMix'''
    batch_size, _, height, width = input1.shape
    ratio = np.zeros([batch_size])

    rx = np.random.uniform(0, height)
    ry = np.random.uniform(0, width)
    rh = np.sqrt(1 - alpha) * height
    rw = np.sqrt(1 - alpha) * width
    x1 = int(np.clip(rx - rh / 2, a_min=0., a_max=height))
    x2 = int(np.clip(rx + rh / 2, a_min=0., a_max=height))
    y1 = int(np.clip(ry - rw / 2, a_min=0., a_max=width))
    y2 = int(np.clip(ry + rw / 2, a_min=0., a_max=width))
    input1[:, :, x1:x2, y1:y2] = input2[:, :, x1:x2, y1:y2]
    ratio += 1 - (x2 - x1) * (y2 - y1) / (width * height)

    ratio = torch.tensor(ratio, dtype=torch.float32)
    if device == 'cuda':
        ratio = ratio.cuda()

    return input1, ratio


def mixup_graph(input1, target_reweighted, saliency1, indices, block_num=2, alpha=0.5, beta=0., gamma=0., eta=0.2,
                neigh_size=2, n_labels=2,
                mean=None, std=None, transport=False, t_eps=10.0, t_size=16, noise=None, adv_mask1=0, adv_mask2=0,
                device='cuda', mp=None,
                scribble1=None, scribble2=None, anchor_lambda=15.0):
    '''Puzzle Mix'''
    input2 = input1[indices].clone()

    batch_size, _, _, width = input1.shape
    block_size = width // block_num
    neigh_size = min(neigh_size, block_size)
    t_size = min(t_size, block_size)

    beta = beta / block_num / 16

    saliency1_pool = F.avg_pool2d(saliency1, block_size)  # [B,1,H,W] or [B,H,W]
    if saliency1.dim() == 4:
        saliency1_pool = saliency1_pool.squeeze(1)
    unary1_torch = saliency1_pool / (saliency1_pool.reshape(batch_size, -1).sum(1, keepdim=True).unsqueeze(-1) + 1e-8)
    unary2_torch = unary1_torch[indices].clone()

    if scribble1 is not None:
        scribble_density1 = F.avg_pool2d((scribble1 != 255).float().unsqueeze(1), block_size)  # [B,1,block,block]
        scribble_density2 = F.avg_pool2d((scribble2 != 255).float().unsqueeze(1), block_size)[indices]
    else:
        scribble_density1 = scribble_density2 = None

    # calculate pairwise terms
    input1_pool = F.avg_pool2d(input1 * std + mean, neigh_size)
    input2_pool = input1_pool[indices]

    pw_x = torch.zeros([batch_size, 2, 2, block_num - 1, block_num], device=device)
    pw_y = torch.zeros([batch_size, 2, 2, block_num, block_num - 1], device=device)

    k = block_size // neigh_size

    pw_x[:, 0, 0], pw_y[:, 0, 0] = neigh_penalty(input2_pool, input2_pool, k)
    pw_x[:, 0, 1], pw_y[:, 0, 1] = neigh_penalty(input2_pool, input1_pool, k)
    pw_x[:, 1, 0], pw_y[:, 1, 0] = neigh_penalty(input1_pool, input2_pool, k)
    pw_x[:, 1, 1], pw_y[:, 1, 1] = neigh_penalty(input1_pool, input1_pool, k)

    pw_x = beta * gamma * pw_x
    pw_y = beta * gamma * pw_y

    # re-define unary and pairwise terms to draw graph
    unary1 = unary1_torch.clone()
    unary2 = unary2_torch.clone()

    unary2[:, :-1, :] += (pw_x[:, 1, 0] + pw_x[:, 1, 1]) / 2.
    unary1[:, :-1, :] += (pw_x[:, 0, 1] + pw_x[:, 0, 0]) / 2.
    unary2[:, 1:, :] += (pw_x[:, 0, 1] + pw_x[:, 1, 1]) / 2.
    unary1[:, 1:, :] += (pw_x[:, 1, 0] + pw_x[:, 0, 0]) / 2.

    unary2[:, :, :-1] += (pw_y[:, 1, 0] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, :-1] += (pw_y[:, 0, 1] + pw_y[:, 0, 0]) / 2.
    unary2[:, :, 1:] += (pw_y[:, 0, 1] + pw_y[:, 1, 1]) / 2.
    unary1[:, :, 1:] += (pw_y[:, 1, 0] + pw_y[:, 0, 0]) / 2.

    pw_x = (pw_x[:, 1, 0] + pw_x[:, 0, 1] - pw_x[:, 1, 1] - pw_x[:, 0, 0]) / 2
    pw_y = (pw_y[:, 1, 0] + pw_y[:, 0, 1] - pw_y[:, 1, 1] - pw_y[:, 0, 0]) / 2

    unary1 = unary1.detach().cpu().numpy()
    unary2 = unary2.detach().cpu().numpy()
    pw_x = pw_x.detach().cpu().numpy()
    pw_y = pw_y.detach().cpu().numpy()

    # solve graphcut or fallback
    if not GCO_AVAILABLE:
        # Fallback: generate a simple binary mask on blocks using alpha as probability
        # mask shape: (batch, block_num, block_num)
        mask = (torch.rand((batch_size, block_num, block_num), device=device) < alpha).float()
        mask = mask.detach().cpu().numpy()
    else:
        if mp is None:
            mask = []
            for i in range(batch_size):
                mask.append(graphcut_multi(unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
        else:
            input_mp = []
            for i in range(batch_size):
                input_mp.append((unary2[i], unary1[i], pw_x[i], pw_y[i], alpha, beta, eta, n_labels))
            mask = mp.starmap(graphcut_multi, input_mp)

    # optimal mask
    mask = torch.tensor(mask, dtype=torch.float32, device=device)
    mask = mask.unsqueeze(1)

    # add adversarial noise
    if adv_mask1 == 1.:
        input1 = input1 * std + mean + noise
        input1 = torch.clamp(input1, 0, 1)
        input1 = (input1 - mean) / std

    if adv_mask2 == 1.:
        input2 = input2 * std + mean + noise[indices]
        input2 = torch.clamp(input2, 0, 1)
        input2 = (input2 - mean) / std

    # transport (skip when fallback mode)
    if transport and GCO_AVAILABLE:
        if t_size == -1:
            t_block_num = block_num
            t_size = block_size
        elif t_size < block_size:
            t_block_num = width // t_size
            mask = F.interpolate(mask, size=t_block_num, mode='nearest')
            saliency1_pool_down = F.avg_pool2d(saliency1, t_size)
            unary1_torch_down = saliency1_pool_down / (
                        saliency1_pool_down.reshape(batch_size, -1).sum(1, keepdim=True).unsqueeze(-1) + 1e-8)
            unary2_torch_down = unary1_torch_down[indices]
            if scribble1 is not None:
                scribble_density1_down = F.avg_pool2d((scribble1 != 255).float().unsqueeze(1), t_size)
                scribble_density2_down = scribble_density1_down[indices]
            else:
                scribble_density1_down = None
                scribble_density2_down = None
        else:
            t_block_num = block_num
            unary1_torch_down = unary1_torch
            unary2_torch_down = unary2_torch
            scribble_density1_down = scribble_density1
            scribble_density2_down = scribble_density2

        # input1
        plan = mask_transport(mask, unary1_torch_down, scribble_density1 is not None and scribble_density1_down or None,
                              eps=t_eps, anchor_lambda=anchor_lambda)
        input1 = transport_image(input1, plan, batch_size, t_block_num, t_size)

        # input2
        plan = mask_transport(1 - mask, unary2_torch_down, scribble2 is not None and scribble_density2_down or None,
                              eps=t_eps, anchor_lambda=anchor_lambda)
        input2 = transport_image(input2, plan, batch_size, t_block_num, t_size)

    # final mask and mixed ratio
    mask = F.interpolate(mask, size=width, mode='nearest')
    ratio = mask.reshape(batch_size, -1).mean(-1)

    # <<< UPGRADE: 实时验证scribble锚定效果（绿色高亮打印）>>>
    if scribble1 is not None and scribble2 is not None:  # 只有开启锚定时才打印
        # 计算每张图的平均scribble像素数（batch平均）
        s1 = (scribble1 != 255).sum(dim=[1, 2]).float().mean().item()
        s2 = (scribble2 != 255).sum(dim=[1, 2]).float().mean().item()

        # 混合后的scribble（考虑mask可能不是严格0/1，但我们用>0.5近似）
        mixed_scribble = mask * scribble1 + (1 - mask) * scribble2
        s_m = (mixed_scribble != 255).sum(dim=[1, 2]).float().mean().item()

        if max(s1, s2) > 0:
            gain_rate = (s_m - max(s1, s2)) / max(s1, s2) * 100
            total_rate = s_m / (s1 + s2) * 100 if (s1 + s2) > 0 else 0
            
        else:
            print("\033[93m[Warning] 本batch无scribble，跳过\033[0m")

    # 原返回语句保持不变
    return mask * input1 + (1 - mask) * input2, ratio, mask


def mask_transport(mask, saliency_pool, scribble_density_pool=None, eps=0.01, anchor_lambda=15.0):
    '''optimal transport plan'''
    batch_size = mask.shape[0]
    block_num = mask.shape[-1]

    n_iter = int(block_num)
    C = cost_matrix_dict[str(block_num)]

    z = (mask > 0).float()
    saliency_pool = saliency_pool.cuda()
    ###maxmize target

    cost = eps * C - saliency_pool.reshape(-1, block_num ** 2, 1) * z.reshape(-1, 1, block_num ** 2)

    if scribble_density_pool is not None:
        scribble_density_pool = scribble_density_pool.cuda()
        cost = cost - anchor_lambda * scribble_density_pool.reshape(-1, block_num ** 2, 1) * z.reshape(-1, 1,
                                                                                                       block_num ** 2)

    # row and col
    for _ in range(n_iter):
        row_best = cost.min(-1)[1]
        plan = torch.zeros_like(cost).scatter_(-1, row_best.unsqueeze(-1), 1)

        # column resolve
        cost_fight = plan * cost
        col_best = cost_fight.min(-2)[1]
        plan_win = torch.zeros_like(cost).scatter_(-2, col_best.unsqueeze(-2), 1) * plan
        plan_lose = (1 - plan_win) * plan

        cost += plan_lose * 999.0

    return plan_win


def transport_image(img, plan, batch_size, block_num, block_size):
    '''apply transport plan to images'''
    # convert channel from 3 to 1
    input_patch = img.reshape([batch_size, 1, block_num, block_size, block_num * block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 1, block_num, block_num, block_size, block_size]).transpose(-2, -1)
    input_patch = input_patch.reshape([batch_size, 1, block_num ** 2, block_size, block_size]).permute(0, 1, 3, 4,
                                                                                                       2).unsqueeze(-1)

    input_transport = plan.transpose(-2, -1).unsqueeze(1).unsqueeze(1).unsqueeze(1).matmul(input_patch).squeeze(
        -1).permute(0, 1, 4, 2, 3)
    input_transport = input_transport.reshape([batch_size, 1, block_num, block_num, block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 1, block_num, block_num * block_size, block_size])
    input_transport = input_transport.transpose(-2, -1).reshape(
        [batch_size, 1, block_num * block_size, block_num * block_size])

    return input_transport