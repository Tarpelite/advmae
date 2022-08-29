import torch
import copy
state_dict_path = "/mnt/tianyu/workspace/adv_seo_clip/mae_pretrain_vit_base.pth"

state_dict = torch.load(state_dict_path, map_location="cpu")
print(state_dict["model"].keys())
keys = state_dict["model"].keys()
new_state_dict = copy.deepcopy(state_dict)
for k in keys:
    new_state_dict["model"]["noise_{}".format(k)] = copy.deepcopy(state_dict["model"][k])
print(new_state_dict["model"].keys())
# print(new_state_dict.keys())
# del new_state_dict["optimizer"]
torch.save(new_state_dict, "/mnt/tianyu/workspace/adv_seo_clip/mae_pretrain_vit_base_noise.pth")
