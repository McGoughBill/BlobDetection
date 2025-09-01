import torch
from typing import List, Union

@torch.no_grad()
def build_yolo_targets_batch(
    labels: List[Union[torch.Tensor, list]],
    anchors: torch.Tensor,
    grid_size: int,
    num_classes: int,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Build YOLO-style targets for a batch.

    Args:
        labels: list of length B; each element is (Ni, 5) with rows [class_id, x, y, w, h],
                all normalized to [0,1] (x,y are centers). Ni can be zero.
        anchors: (A, 2) tensor of [aw, ah] normalized to [0,1].
        grid_size: S (e.g., 7 means an SxS grid).
        num_classes: number of classes C.

    Returns:
        targets: (B, A, S, S, 5 + C) tensor.
                 For a matched (b, a, i, j):
                   tx = x*S - j
                   ty = y*S - i
                   tw = log(w / aw)
                   th = log(h / ah)
                   obj = 1
                   class one-hot at offset 5.
                 Unassigned entries are zeros.
    """
    eps = 1e-9

    # Ensure torch tensors and consistent device/dtype
    if not isinstance(anchors, torch.Tensor):
        anchors = torch.tensor(anchors, dtype=torch.float32)
    anchors = anchors.to(dtype=torch.float32)
    device = anchors.device
    A = anchors.shape[0]
    S = int(grid_size)
    C = int(num_classes)
    B = len(labels)

    targets = torch.zeros((B, A, 5 + C, S, S, ), device=device, dtype=torch.float32)
    # Track best IoU used at each (b,a,i,j) so we can resolve collisions by IoU
    best_iou_map = torch.zeros((B, A, S, S), device=device, dtype=torch.float32)

    # Helper: IoU for width/height pairs assuming same center
    # wh1: (A,2), wh2: (N,2) -> (A,N)
    def iou_wh(wh1: torch.Tensor, wh2: torch.Tensor) -> torch.Tensor:
        wh1 = wh1.unsqueeze(1)        # (A,1,2)
        wh2 = wh2.unsqueeze(0)        # (1,N,2)
        inter = torch.minimum(wh1, wh2).prod(dim=-1)            # (A,N)
        area1 = wh1.prod(dim=-1)                                 # (A,1)
        area2 = wh2.prod(dim=-1)                                 # (1,N)
        union = area1 + area2 - inter
        return inter / (union + eps)

    for b, lab in enumerate(labels):
        if lab is None:
            continue

        if not isinstance(lab, torch.Tensor):
            lab = torch.tensor(lab, dtype=torch.float32, device=device)
        else:
            lab = lab.to(device=device, dtype=torch.float32)

        if lab.numel() == 0:
            continue

        # Expect [class, x, y, w, h]
        assert lab.shape[-1] == 5, "Each label row must be [class, x, y, w, h]."

        cls = lab[:, 0].to(dtype=torch.long).clamp(min=0, max=C-1)
        # Clamp coords/size into (0,1) to avoid edge issues
        xywh = lab[:, 1:5].clamp(min=0.0, max=1.0 - 1e-6)
        x, y, w, h = xywh.unbind(-1)

        # Which cell (i=row, j=col)
        gx = x * S
        gy = y * S
        j = gx.floor().to(torch.long).clamp_(0, S - 1)
        i = gy.floor().to(torch.long).clamp_(0, S - 1)

        # Offsets (0..1)
        tx = gx - j.to(torch.float32)
        ty = gy - i.to(torch.float32)

        # Choose best anchor for each GT by IoU on wh
        gt_wh = torch.stack([w.clamp_min(eps), h.clamp_min(eps)], dim=1)  # (N,2)
        ious = iou_wh(anchors.clamp_min(eps), gt_wh)  # (A,N)
        best_a = ious.argmax(dim=0)                   # (N,)
        best_a_iou = ious.gather(0, best_a.unsqueeze(0)).squeeze(0)  # (N,)

        # Place targets; if collision, keep the assignment with higher IoU
        for n in range(lab.shape[0]):
            a = int(best_a[n].item())
            ii = int(i[n].item())
            jj = int(j[n].item())

            prev_iou = best_iou_map[b, a, ii, jj]
            cur_iou = best_a_iou[n]
            if cur_iou > prev_iou:
                # Box size params as log(w/aw), log(h/ah)
                tw = torch.log(gt_wh[n, 0] / (anchors[a, 0] + eps) + eps)
                th = torch.log(gt_wh[n, 1] / (anchors[a, 1] + eps) + eps)

                # Zero out any previous class one-hot (helps when overwriting)
                targets[b, a, 5:, ii, jj] = 0.0
                targets[b, a, 1, ii, jj] = tx[n]
                targets[b, a, 2, ii, jj] = ty[n]
                targets[b, a, 3,  ii, jj] = tw
                targets[b, a, 4, ii, jj] = th
                targets[b, a, 0, ii, jj] = 1.0  # objectness
                targets[b, a, 5+ int(cls[n].item()), ii, jj] = 1.0

                best_iou_map[b, a, ii, jj] = cur_iou

    return targets.to(device)
