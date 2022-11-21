import requests
import pickle 
import torch
import timm
import torchvision

# torch.hub.set_dir("/srv/tempdd/tmaho/models/")


def get_model(model_name, jpeg_module=False, preload_model=None):
    # torch.hub.set_dir("/srv/tempdd/tmaho/torch_models")
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalizer = torchvision.transforms.Normalize(mean=mean, std=std)
    if preload_model:
        import copy
        normalizer, model = copy.deepcopy(preload_model)
    elif model_name == "madry":
        model, std, mean = load_madry()
    elif model_name.lower().startswith("torchvision"):
        model = getattr(torchvision.models, model_name[len("torchvision_"):])(pretrained=True)
    else:
        model = timm.create_model(model_name, pretrained=True)
        mean = model.default_cfg["mean"]
        std = model.default_cfg["std"]
        normalizer = torchvision.transforms.Normalize(mean=mean, std=std)


    model = torch.nn.Sequential(
                    normalizer,
                    model
                )

    model = model.eval()
    return model


def load_madry():
    import dill
    # load from https://download.pytorch.org/models/resnet50-19c8e357.pth
    weights_path = "/nfs/nas4/bbonnet/bbonnet/thibault/extra_model/imagenet_l2_3_0.pt"
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'), pickle_module=dill)
    sd = checkpoint["model"]
    for w in ["module.", "attacker.", "model."]:
        sd = {k.replace(w, ""):v for k,v in sd.items()}

    std = sd["normalize.new_std"].flatten()
    mean = sd["normalize.new_mean"].flatten()
    
    del sd["normalize.new_std"]
    del sd["normalize.new_mean"]
    del sd["normalizer.new_std"]
    del sd["normalizer.new_mean"]

    model = torchvision.models.resnet50(pretrained=False)
    model.load_state_dict(sd)
    return model, std, mean

