from torch import nn
from torch.nn import functional as F


class Conv1d(nn.Conv1d):
    """
        Extend nn.Conv1d for incremental dilated convolutions
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clear_buffer()
        self._linearized_weight = None
        self.register_backward_hook(self._clear_linearzied_weight)
    
    def clear_buffer(self):
        self.input_buffer = None

    def _clear_linearzied_weight(self, *args):
        self._linearized_weight = None

    def incremental_forward(self, input):
        # input: (B, T, C)
        if self.training:
            raise RuntimeError('incremental_forward only supports eval mode')
        
        # run forward pre hooks (e.g., weight norm)
        for hook in self._forward_pre_hooks.values():
            hook(self, input)

        # reshape weight
        weight = self._get_linearized_weight()
        kw = self.kernel_size[0]
        dilation = self.dilation[0]

        batch_size = input.size(0)  # input: batch_size x num_timestep x num_channel
        if kw > 1:
            input = input.data
            if self.input_buffer is None:
                self.input_buffer = input.new(batch_size, kw + (kw - 1) * (dilation - 1), input.size(2))
            else:
                # shift buffer
                self.input_buffer[:, :-1, :] = self.input_buffer[:, 1:, :].clone()
            # append next input
            self.input_buffer[:, -1, :] = input[:, -1, :]
            input = self.input_buffer
            if dilation > 1:
                input = input[:, 0::dilation, :].contiguous()
        output = F.linear(input.view(batch_size, -1), weight, self.bias)
        return output.view(batch_size, 1, -1)

    def _get_linearized_weight(self):
        if self._linearized_weight is None:
            kw = self.kernel_size[0]
            # nn.Conv1d
            if self.weight.size() == (self.out_channels, self.in_channels, kw):
                weight = self.weight.transpose(1, 2).contiguous()
            else:
                # fairseq.modules.conv_tbc.ConvTBC
                weight = self.weight.transpose(2, 1).transpose(1, 0).contiguous()
            assert weight.size() == (self.out_channels, kw, self.in_channels)
            self._linearized_weight = weight.view(self.out_channels, -1)
        return self._linearized_weight