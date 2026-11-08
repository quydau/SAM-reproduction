"""Bước train giả lập nhiều GPU, chỉ dùng cho thực nghiệm label noise (§3.3).

Bài báo (§3, trang 5): "we train all models on a single host having 8 Nvidia
V100 GPUs. To compute the SAM update when parallelizing across multiple
accelerators, we divide each data batch evenly among the accelerators,
independently compute the SAM gradient on each accelerator, and average the
resulting sub-batch SAM gradients to obtain the final SAM update."

Với batch 256 trên 8 GPU, mỗi GPU nhận 32 ảnh. File này giả lập điều đó trên
một GPU: chia mỗi batch thành ``num_micro_batches`` micro-batch, mỗi micro-batch
tính gradient (và nhiễu SAM) riêng, rồi lấy trung bình.

File này tách riêng để không sửa ``train_utils.py`` / ``optimizers.py`` dùng
chung. ``use_micro_batch_train_step`` tạm thay ``train_utils.train_step`` bằng
bước train ở đây trong lúc chạy một run, rồi trả lại bản gốc.
"""

from contextlib import contextmanager

import torch

from sam_pytorch import train_utils
from sam_pytorch.optimizers import SAM
from sam_pytorch.train_utils import (_add_l2_gradient, _get_rng_state, _loss,
                                     _move_batch_to_model_device, _set_rng_state,
                                     _unpack_batch, error_rate_metric)


def _split_into_micro_batches(device_batches, num_micro_batches):
    """Chia mỗi batch (images, labels, mask) thành ``num_micro_batches`` phần bằng nhau."""
    micro_batches = []
    for images, labels, mask in device_batches:
        batch_size = images.shape[0]
        if batch_size % num_micro_batches != 0:
            raise ValueError(
                f"batch of {batch_size} cannot be split into {num_micro_batches} "
                "equal micro-batches; use a divisible batch_size (drop_last=True)")
        image_parts = images.chunk(num_micro_batches)
        label_parts = labels.chunk(num_micro_batches)
        mask_parts = (mask.chunk(num_micro_batches) if mask is not None
                      else [None] * num_micro_batches)
        micro_batches.extend(zip(image_parts, label_parts, mask_parts))
    return micro_batches


@torch.no_grad()
def _remove_sam_perturbation(optimizer: SAM) -> None:
    """Đưa trọng số từ ``w + e`` về lại ``w`` (không cập nhật).

    ``SAM.first_step`` lưu ``w`` vào ``optimizer.state[p]["old_p"]``. Lấy ra và
    xóa luôn, để micro-batch sau không dùng nhầm giá trị cũ.
    """
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if "old_p" in optimizer.state[parameter]:
                parameter.data = optimizer.state[parameter].pop("old_p")


def train_step_per_micro_batch(model, optimizer, batches, *, num_micro_batches,
                               label_smoothing=0.0, l2_reg=None,
                               gradient_clip_norm=None, no_weight_decay_on_bn=False,
                               **unused_kwargs):
    """Một bước cập nhật, giả lập ``num_micro_batches`` GPU.

    Với mỗi micro-batch, giống một GPU riêng:
      1. BN bắt đầu từ cùng running stats; BN dùng thống kê của micro-batch đó.
      2. Gradient tại w (cộng gradient L2).
      3. SAM: nhiễu riêng e_i = rho * g_i / ||g_i||, gradient tại w + e_i
         (cộng gradient L2), rồi gỡ nhiễu về w.
         SGD: giữ gradient ở bước 2.
    Sau đó: trung bình gradient các micro-batch, trung bình running stats BN
    (như đồng bộ giữa các GPU), clip, rồi cập nhật một lần.
    """
    is_sam = isinstance(optimizer, SAM)
    l2_reg = float(l2_reg or 0.0)
    if is_sam:
        # SAM.first_step / add_l2_gradient đọc hệ số L2 từ param group.
        for group in optimizer.param_groups:
            group["l2_reg"] = l2_reg
            group["l2_skip_1d"] = no_weight_decay_on_bn

    device_batches = [_move_batch_to_model_device(model, *_unpack_batch(b)) for b in batches]
    micro_batches = _split_into_micro_batches(device_batches, num_micro_batches)
    num_parts = len(micro_batches)

    parameters = [p for p in model.parameters() if p.requires_grad]
    buffers = list(model.buffers())
    gradient_sums = [torch.zeros_like(p) for p in parameters]
    buffers_at_start = [b.detach().clone() for b in buffers]
    buffer_sums = [torch.zeros_like(b) if b.is_floating_point() else None for b in buffers]
    buffers_after_first_pass = buffers_at_start
    loss_sum = error_sum = 0.0

    for images, labels, mask in micro_batches:
        # 1. Mọi "GPU" bắt đầu từ cùng BN running stats.
        with torch.no_grad():
            for buffer, start_value in zip(buffers, buffers_at_start):
                buffer.copy_(start_value)

        # 2. Gradient tại w.
        optimizer.zero_grad(set_to_none=True)
        rng_state = _get_rng_state()
        loss, logits = _loss(model, images, labels, mask, label_smoothing)
        loss.backward()
        # Chỉ lượt forward đầu được cập nhật BN running stats.
        with torch.no_grad():
            buffers_after_first_pass = [b.detach().clone() for b in buffers]

        if is_sam:
            # 3. Nhiễu riêng của micro-batch này, gradient tại w + e_i, rồi gỡ nhiễu.
            optimizer.first_step(zero_grad=True)  # tự cộng gradient L2 trước khi tính e_i
            _set_rng_state(rng_state)
            loss, logits = _loss(model, images, labels, mask, label_smoothing)
            loss.backward()
            optimizer.add_l2_gradient()
            _remove_sam_perturbation(optimizer)
        else:
            _add_l2_gradient(model, l2_reg, no_weight_decay_on_bn)

        with torch.no_grad():
            for gradient_sum, parameter in zip(gradient_sums, parameters):
                if parameter.grad is not None:
                    gradient_sum.add_(parameter.grad)
            for buffer_sum, value in zip(buffer_sums, buffers_after_first_pass):
                if buffer_sum is not None:
                    buffer_sum.add_(value)
        loss_sum = loss_sum + loss.detach() / num_parts
        error_sum = error_sum + error_rate_metric(logits.detach(), labels, mask) / num_parts

    # Trung bình gradient và BN running stats qua các micro-batch.
    with torch.no_grad():
        for parameter, gradient_sum in zip(parameters, gradient_sums):
            parameter.grad = gradient_sum / num_parts
        for buffer, buffer_sum, after_first_pass in zip(buffers, buffer_sums,
                                                       buffers_after_first_pass):
            if buffer_sum is not None:
                buffer.copy_(buffer_sum / num_parts)
            else:
                buffer.copy_(after_first_pass)  # num_batches_tracked: +1 mỗi bước

    clip_limit = (gradient_clip_norm if gradient_clip_norm and gradient_clip_norm > 0
                  else float("inf"))
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_limit)
    grad_norm = total_norm * (clip_limit / (total_norm + 1e-6)).clamp(max=1.0)
    if is_sam:
        optimizer.base_optimizer.step()  # trọng số đã ở w, chỉ cần cập nhật
    else:
        optimizer.step()

    with torch.no_grad():
        param_norm = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm([p.detach() for p in model.parameters()])))
    return {
        "train_loss": loss_sum,
        "train_error_rate": error_sum,
        "gradient_norm": grad_norm,
        "param_norm": param_norm,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
    }


@contextmanager
def use_micro_batch_train_step(num_micro_batches: int):
    """Trong khối ``with``, ``train_utils.train`` dùng bước train giả lập nhiều GPU.

    ``num_micro_batches <= 1`` thì không đổi gì (dùng ``train_step`` gốc).
    """
    if num_micro_batches <= 1:
        yield
        return

    original_train_step = train_utils.train_step

    def micro_batch_train_step(model, optimizer, batches, **kwargs):
        return train_step_per_micro_batch(
            model, optimizer, batches, num_micro_batches=num_micro_batches, **kwargs)

    train_utils.train_step = micro_batch_train_step
    try:
        yield
    finally:
        train_utils.train_step = original_train_step
