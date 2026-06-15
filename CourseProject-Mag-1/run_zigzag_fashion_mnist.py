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
    zigzag.pipelines.Params(k_neighbors=2, dimension=3),
    zigzag.pipelines.Params(k_neighbors=3, dimension=3),
    zigzag.pipelines.Params(k_neighbors=4, dimension=3),
    zigzag.pipelines.Params(k_neighbors=5, dimension=2),
    zigzag.pipelines.Params(k_neighbors=7, dimension=1),
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
    train_ds = torchvision.datasets.FashionMNIST("fashion_mnist", train=True, download=True, transform=transforms)
    test_ds = torchvision.datasets.FashionMNIST("fashion_mnist", train=False, download=True, transform=transforms)

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

    dumper = zigzag.utils.UniversalDumper(f"zigzag_results/fashion_mnist/{model_name}")

    pretrained_dumper = dumper.make_subdumper("pretrained")
    model, _ = make_model(torch.nn.Identity())
    zigzag.pipelines.validate_pretrained(model, train_ds, train_y, test_ds, test_y, pretrained_dumper)
    if use_bulk:
        zigzag.pipelines.analyze_bulk(model, train_ds, PARAMS, pretrained_dumper, class_labels=train_y)
    else:
        hidden_states = pretrained_dumper.execute(zigzag.nn.collect_hidden_states, "hidden_states", model, train_ds)
        zigzag.pipelines.analyze(hidden_states, PARAMS, pretrained_dumper, class_labels=train_y)

    if only_pretrained:
        return

    finetuned_dumper = dumper.make_subdumper("finetuned")
    model, _ = make_model(pretrained_dumper.get_dump("trained_head"))
    zigzag.pipelines.train_validate(model, train_ds, test_ds, dumper, learning_rate=1e-5)
    if use_bulk:
        zigzag.pipelines.analyze_bulk(model, train_ds, PARAMS, finetuned_dumper, class_labels=train_y)
    else:
        hidden_states = dumper.execute(zigzag.nn.collect_hidden_states, "hidden_states", model, train_ds)
        zigzag.pipelines.analyze(hidden_states, PARAMS, finetuned_dumper, class_labels=train_y)

def make_vit_b_16(head: torch.nn.Module):
    model = torchvision.models.vit_b_16(num_classes=1000, weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    model.heads = head
    return model, torchvision.models.ViT_B_16_Weights.DEFAULT.transforms

def make_vit_b_32(head: torch.nn.Module):
    model = torchvision.models.vit_b_32(num_classes=1000, weights=torchvision.models.ViT_B_32_Weights.DEFAULT)
    model.heads = head
    return model, torchvision.models.ViT_B_32_Weights.DEFAULT.transforms

def make_vit_h_14(head: torch.nn.Module):
    model = torchvision.models.vit_h_14(num_classes=1000, weights=torchvision.models.ViT_H_14_Weights.DEFAULT)
    model.heads = head
    return model, torchvision.models.ViT_H_14_Weights.DEFAULT.transforms

def make_vit_l_16(head: torch.nn.Module):
    model = torchvision.models.vit_l_16(num_classes=1000, weights=torchvision.models.ViT_L_16_Weights.DEFAULT)
    model.heads = head
    return model, torchvision.models.ViT_L_16_Weights.DEFAULT.transforms

def make_vit_l_32(head: torch.nn.Module):
    model = torchvision.models.vit_l_32(num_classes=1000, weights=torchvision.models.ViT_L_32_Weights.DEFAULT)
    model.heads = head
    return model, torchvision.models.ViT_L_32_Weights.DEFAULT.transforms

def make_resnet18(head: torch.nn.Module):
    model = torchvision.models.resnet18(num_classes=1000, weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = head
    return model, torchvision.models.ResNet18_Weights.DEFAULT.transforms

def make_resnet34(head: torch.nn.Module):
    model = torchvision.models.resnet34(num_classes=1000, weights=torchvision.models.ResNet34_Weights.DEFAULT)
    model.fc = head
    return model, torchvision.models.ResNet34_Weights.DEFAULT.transforms

def make_resnet50(head: torch.nn.Module):
    model = torchvision.models.resnet50(num_classes=1000, weights=torchvision.models.ResNet50_Weights.DEFAULT)
    model.fc = head
    return model, torchvision.models.ResNet50_Weights.DEFAULT.transforms

def make_resnet101(head: torch.nn.Module):
    model = torchvision.models.resnet101(num_classes=1000, weights=torchvision.models.ResNet101_Weights.DEFAULT)
    model.fc = head
    return model, torchvision.models.ResNet101_Weights.DEFAULT.transforms

def make_resnet152(head: torch.nn.Module):
    model = torchvision.models.resnet152(num_classes=1000, weights=torchvision.models.ResNet152_Weights.DEFAULT)
    model.fc = head
    return model, torchvision.models.ResNet152_Weights.DEFAULT.transforms


run_model(make_vit_b_16, "vit_b_16", use_bulk = False)
run_model(make_vit_b_32, "vit_b_32", use_bulk = False)
# run_model(make_vit_h_14, "vit_h_14", use_bulk = False)
run_model(make_vit_l_16, "vit_l_16", use_bulk = False, only_pretrained = True)
# run_model(make_vit_l_32, "vit_l_32", use_bulk = False, only_pretrained = True)
# run_model(make_resnet18, "resnet18", subset = 1/6)
# run_model(make_resnet34, "resnet34", subset = 1/6)
# run_model(make_resnet50, "resnet50", subset = 1/6)
# run_model(make_resnet101, "resnet101", subset = 1/6)
# run_model(make_resnet152, "resnet152", subset = 1/6)
