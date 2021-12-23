import torch
import torchvision
import numpy as np

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import LinalgOnTensorsTosaBackend
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend


def iou_of(bbox1, bbox2):
    """get iou of two boxes.
    Args:
        bbox1 array(n, 4): (x1, y1, x2, y2)
        bbox2 array(n, 4): (x1, y1, x2, y2)

    return: value of iou
    """
    area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
    area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
    lt = np.maximum(bbox1[:, :2], bbox2[:, :2])
    rb = np.minimum(bbox1[:, 2:], bbox2[:, 2:])

    overlap_coord = (rb - lt).clip(0)
    overlap = overlap_coord[:, 0] * overlap_coord[:, 1]
    union = area1 + area2 - overlap

    return overlap / union

class IouOfModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.fc0 = torch.nn.Linear(3, 5)
        self.tanh0 = torch.nn.Tanh()

    @export
    @annotate_args([
        None,
        ([1024, 4], torch.float32, True),
        ([1024, 4], torch.float32, True),
    ])
    def forward(self, bbox1, bbox2):
        area1 = (bbox1[:, 2] - bbox1[:, 0]) * (bbox1[:, 3] - bbox1[:, 1])
        area2 = (bbox2[:, 2] - bbox2[:, 0]) * (bbox2[:, 3] - bbox2[:, 1])
        # area1 = torch.mul(torch.sub(bbox1[:, 2], bbox1[:, 0]), torch.sub(bbox1[:, 3], bbox1[:, 1]))
        # area2 = torch.mul(torch.sub(bbox2[:, 2], bbox2[:, 0]), torch.sub(bbox2[:, 3], bbox2[:, 1]))
        lt = torch.maximum(bbox1[:, :2], bbox2[:, :2])
        rb = torch.minimum(bbox1[:, 2:], bbox2[:, 2:])

        overlap_coord = (rb - lt).clip(0)
        overlap = overlap_coord[:, 0] * overlap_coord[:, 1]
        union = area1 + area2 - overlap

        return overlap / union
        # return area1 + area2


# BACKEND = LinalgOnTensorsTosaBackend()
BACKEND = RefBackendLinalgOnTensorsBackend()


def compile_module(program: torch.nn.Module):
    """Compiles a torch.nn.Module into an compiled artifact.

    This artifact is suitable for inclusion in a user's application. It only
    depends on the rebackend runtime.
    """
    ## Script the program.
    scripted = torch.jit.script(program)
    print("-------------------------------graph-------------------------------")
    print(scripted.graph)
    print("-------------------------------code-------------------------------")
    print(scripted.code)

    ## Extract annotations.
    class_annotator = ClassAnnotator()
    extract_annotations(program, scripted, class_annotator)

    ## Import the TorchScript module into MLIR.
    mb = ModuleBuilder()
    mb.import_module(scripted._c, class_annotator)
    print("-------------------------------init IR-------------------------------")
    mb.module.dump()

    ## Lower the MLIR from TorchScript to RefBackend, passing through linalg-on-tensors.
    pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline', mb.module.context)
    pm.run(mb.module)
    print("-------------------------------torch-back-pipeline-------------------------------")
    mb.module.dump()

    pm = PassManager.parse('torch-backend-to-linalg-on-tensors-backend-pipeline', mb.module.context)
    # pm = PassManager.parse('builtin.func(convert-torch-to-linalg)', mb.module.context)
    # pm = PassManager.parse('torch-verify-invariants-before-backend-lowering', mb.module.context)
    pm.run(mb.module)
    print("-------------------------------linalg-------------------------------")
    mb.module.dump()

    ## Invoke RefBackend to compile to compiled artifact form.
    compiled_module = BACKEND.compile(mb.module)
    print("-------------------------------after-compile-------------------------------")
    compiled_module.dump()
    return compiled_module


if __name__ == '__main__':
    print("---------------main-------------------")
    mlp_module = IouOfModule()
    # Create the module and compile it.
    compiled = compile_module(mlp_module)
    # compiled.dump()
    # Loads the compiled artifact into the runtime
    jit_module = BACKEND.load(compiled)
    # Run it!
    jit_module.forward(torch.rand(1024, 4).numpy(), torch.rand(1024, 4).numpy())
