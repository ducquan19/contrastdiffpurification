r"""
Read a Lightning checkpoint and save its weights only (a plain ``state_dict``).

Usage:
Show the structure of the checkpoint
```
python -m scripts.convert_lightning_checkpoint --ckpt_path=path/to/checkpoint.ckpt -v
```

remove the prefix of the checkpoint
```
python -m scripts.convert_lightning_checkpoint --ckpt_path=path/to/checkpoint.ckpt -x model --out_path=path/to/output.pt
```
"""

import argparse
import logging

import torch

from src.utils import setup_logging

logger = logging.getLogger(__name__)


class TreeNode:
    """Prefix tree of dotted parameter names, printed to show a checkpoint's module structure."""

    def __init__(self, name):
        self.name = name
        self.children = {}

    def add_child(self, child_name):
        if child_name in self.children:
            return self.children[child_name]
        else:
            new_child = TreeNode(child_name)
            self.children[child_name] = new_child
            return new_child

    def build_tree(self, path):
        current_node = self
        for node_name in path.split('.'):
            current_node = current_node.add_child(node_name)

    def __str__(self, indent=0, max_indent=-1):
        if max_indent > 0 and indent > max_indent:
            return ''
        result = '  ' * indent + self.name + '\n'
        for child in self.children.values():
            result += child.__str__(indent + 1, max_indent)
        return result


_args = argparse.ArgumentParser(
    __file__, description="Read the lightning checkpoint and save weight only."
)

_args.add_argument(
    "--ckpt_path",
    type=str,
    required=True,
    help="The path of the checkpoint file."
)
_args.add_argument(
    "--out_path",
    type=str,
    default=None,
    help="The path of the output file."
)
_args.add_argument(
    "-x",
    "--remove_prefix",
    type=str,
    default="",
    help="Remove the model prefix in the weight name."
)
_args.add_argument(
    "-v",
    "--show_prefix_depth",
    type=int,
    default=0,
    help="Show the depth of the model prefix in the weight name."
)


def main():
    args = _args.parse_args()
    setup_logging()

    lightning_checkpoint = torch.load(args.ckpt_path, map_location=torch.device('cpu'))
    try:
        state_dict = lightning_checkpoint['state_dict']  # type: dict
        logger.info("checkpoint keys: %s", list(lightning_checkpoint.keys()))
    except Exception:
        state_dict = lightning_checkpoint
    logger.info("loaded state dict with %d tensors", len(state_dict))

    if args.show_prefix_depth > 0:
        mkey = TreeNode("")
        for k in state_dict.keys():
            mkey.build_tree(k)
        logger.info("parameter prefixes:\n%s", mkey.__str__(0, args.show_prefix_depth))

    if args.out_path is None:
        logger.info("no --out_path given; nothing saved")
        return

    if args.remove_prefix != "":
        remove_prefix = args.remove_prefix
        if not remove_prefix.endswith("."):
            remove_prefix += "."

        logger.info("removing prefix '%s'", remove_prefix)
        klen = len(remove_prefix)
        for old_k in list(state_dict.keys()):
            if not old_k.startswith(remove_prefix):
                continue
            k = old_k[klen:]
            logger.debug("%s => %s", old_k, k)
            state_dict[k] = state_dict.pop(old_k)

    torch.save(state_dict, args.out_path)
    logger.info("state dict saved to %s", args.out_path)


if __name__ == "__main__":
    main()
