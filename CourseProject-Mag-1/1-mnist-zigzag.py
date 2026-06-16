import sys
sys.path.append("../Diploma/src")

import typing

import torch
import torchvision
import sklearn.model_selection
import torchvision.transforms.v2

import zigzag.nn
import zigzag.utils
import zigzag.pipelines

PARAMS = [
    zigzag.pipelines.Params(k_neighbors=4, dimension=3),
]

def run_model(
    make_model: typing.Callable[[torch.nn.Module], torch.nn.Module],
    model_name: str,
    *,
    use_bulk: bool = True,
    only_pretrained: bool = False,
    subset: typing.Optional[float] = None
):
    _, transforms = make_model(torch.nn.Identity())
    transforms = torchvision.transforms.v2.Compose([torchvision.transforms.v2.Grayscale(num_output_channels=3), transforms()])
    train_ds = torchvision.datasets.MNIST("mnist", train=True, download=True, transform=transforms)
    test_ds = torchvision.datasets.MNIST("mnist", train=False, download=True, transform=transforms)

    train_y = train_ds.targets
    test_y = test_ds.targets
    if subset is not None:
        train_idxs, _ = sklearn.model_selection.train_test_split(
            list(range(len(train_ds))), stratify=train_y, test_size=(1-subset), random_state=42
        )
        test_idxs, _ = sklearn.model_selection.train_test_split(
            list(range(len(test_ds))), stratify=test_y, test_size=(1-subset), random_state=42
        )
        train_y = train_ds.targets[train_idxs]
        test_y = test_ds.targets[test_idxs]
        train_ds = torch.utils.data.Subset(train_ds, indices=train_idxs)
        test_ds = torch.utils.data.Subset(test_ds, indices=test_idxs)

    dumper = zigzag.utils.UniversalDumper(f"zigzag_results/mnist/{model_name}")

    model, _ = make_model(torch.nn.Identity())
    pretrained_dumper = dumper.make_subdumper("pretrained")
    zigzag.pipelines.validate_pretrained(model, train_ds, train_y, test_ds, test_y, pretrained_dumper)
    if not only_pretrained:
        finetuned_dumper = dumper.make_subdumper("finetuned")
        model, _ = make_model(pretrained_dumper.get_dump("trained_head"))
        zigzag.pipelines.train_validate(model, train_ds, test_ds, finetuned_dumper, learning_rate=1e-5)

    if use_bulk:
        zigzag.pipelines.analyze_bulk(model, train_ds, PARAMS, dumper)
    else:
        hidden_states = dumper.execute(zigzag.nn.collect_hidden_states, "hidden_states", model, train_ds)
        zigzag.pipelines.analyze(hidden_states, PARAMS, dumper)

def make_vit_b_32(head: torch.nn.Module):
    model = torchvision.models.vit_b_32(num_classes=1000, weights=torchvision.models.ViT_B_32_Weights.DEFAULT)
    model.heads = head
    return model, torchvision.models.ViT_B_32_Weights.DEFAULT.transforms

def make_vit_l_16(head: torch.nn.Module):
    model = torchvision.models.vit_l_16(num_classes=1000, weights=torchvision.models.ViT_L_16_Weights.DEFAULT)
    model.heads = head
    return model, torchvision.models.ViT_L_16_Weights.DEFAULT.transforms

def make_resnet34(head: torch.nn.Module):
    model = torchvision.models.resnet34(num_classes=1000, weights=torchvision.models.ResNet34_Weights.DEFAULT)
    model.fc = head
    return model, torchvision.models.ResNet34_Weights.DEFAULT.transforms

run_model(make_vit_b_32, "vit_b_32", use_bulk = False)
run_model(make_vit_l_16, "vit_l_16", use_bulk = False, only_pretrained = True)
run_model(make_resnet34, "resnet34", subset = 1/6)
