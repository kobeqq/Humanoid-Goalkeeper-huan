import torch

def inspect_pt(path, max_items=100):
    data = torch.load(path, map_location="cpu")

    print(f"\n=== Loaded: {path} ===")
    print(f"Type: {type(data)}\n")

    def recurse(obj, prefix=""):
        if isinstance(obj, dict):
            print(f"{prefix}dict with {len(obj)} keys")
            for k in obj:
                print(f"{prefix}  [{k}]")
                recurse(obj[k], prefix + "    ")

        elif isinstance(obj, list):
            print(f"{prefix}list len={len(obj)}")
            for i, v in enumerate(obj[:max_items]):
                print(f"{prefix}  [{i}]")
                recurse(v, prefix + "    ")

        elif torch.is_tensor(obj):
            print(f"{prefix}Tensor shape={tuple(obj.shape)}, dtype={obj.dtype}")
            print(f"{prefix}  min={obj.min().item():.4f}, max={obj.max().item():.4f}")

        else:
            print(f"{prefix}{type(obj)}: {obj}")

    recurse(data)

if __name__ == "__main__":
    path = "/home/huan/Humanoid-Goalkeeper-huan/legged_gym/resources/datasets/goalkeeper_from_pkl/goalkeeper_from_pkl.pt"  # 改成你的路径
    inspect_pt(path)