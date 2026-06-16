import typing

import torch
import torchvision
import cvtda.logging
import torch.utils.data
import cvtda.neural_network


@torch.no_grad()
def yield_hidden_states_vit(
    model: torchvision.models.VisionTransformer, x: torch.Tensor, device: torch.device
) -> typing.Generator[torch.Tensor, None, None]:
    x = model._process_input(x.to(device))

    batch_class_token = model.class_token.expand(x.shape[0], -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)
    del batch_class_token

    torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
    x = (x + model.encoder.pos_embedding).cpu()
    yield x[:, -1, :].clone()

    for layer in model.encoder.layers:
        x = layer(x.to(device)).cpu()
        yield x[:, -1, :].clone()


@torch.no_grad()
def yield_hidden_states_resnet(
    model: torchvision.models.ResNet, x: torch.Tensor, device: torch.device
) -> typing.Generator[torch.Tensor, None, None]:
    @torch.no_grad()
    def run_block_no_relu(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if type(block) == torchvision.models.resnet.BasicBlock:
            identity = x

            out = block.conv1(x)
            out = block.bn1(out)
            out = block.relu(out)

            out = block.conv2(out)
            out = block.bn2(out)

            if block.downsample is not None:
                identity = block.downsample(x)

            out += identity
        elif type(block) == torchvision.models.resnet.Bottleneck:
            identity = x

            out = block.conv1(x)
            out = block.bn1(out)
            out = block.relu(out)

            out = block.conv2(out)
            out = block.bn2(out)
            out = block.relu(out)

            out = block.conv3(out)
            out = block.bn3(out)

            if block.downsample is not None:
                identity = block.downsample(x)

            out += identity
        else:
            assert False, f"Unknown resnet block: {type(block)}"
        return out

    x = model.conv1(x.to(device))
    x = model.bn1(x)
    hidden_state = model.maxpool(x.clone()).cpu()
    x = model.relu(x)
    x = model.maxpool(x).cpu()
    yield hidden_state
    del hidden_state
    x = x.to(device)

    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            x = run_block_no_relu(block, x).cpu()
            yield x.clone()
            x = block.relu(x.to(device))


@torch.no_grad()
def yield_hidden_states_batch(
    model: torch.nn.Module, data: torch.Tensor, device: torch.device
) -> typing.Generator[torch.Tensor, None, None]:
    if isinstance(model, torchvision.models.vision_transformer.VisionTransformer):
        return yield_hidden_states_vit(model, data, device)
    if isinstance(model, torchvision.models.ResNet):
        return yield_hidden_states_resnet(model, data, device)
    assert False, f"{type(model)} is not supported"


@torch.no_grad()
def yield_hidden_states(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device = cvtda.neural_network.default_device,
) -> typing.Generator[torch.Tensor, None, None]:
    model = model.to(device).eval()
    data = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)
    data = cvtda.logging.logger().pbar(data, desc=f"Create generators")
    generators = [yield_hidden_states_batch(model, X, device) for X, *_ in data]

    layer_num = 0
    while True:
        layer_num += 1
        result = []
        for batch in cvtda.logging.logger().pbar(generators, desc=f"Collect hidden states, layer {layer_num}"):
            try:
                result.append(next(batch))
            except StopIteration:
                return
        result = torch.concat(result)
        yield result


@torch.no_grad()
def collect_hidden_states(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device = cvtda.neural_network.default_device,
) -> typing.List[torch.Tensor]:
    return list(yield_hidden_states(model, dataset, device))
