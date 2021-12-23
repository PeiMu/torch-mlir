import torch
import torchvision
import numpy as np

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import LinalgOnTensorsTosaBackend
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend


def match_score(x_a, x_b, y_a, y_b, offset_a, offset_b, one):
    # x_a, y_a = point_a
    # x_b, y_b = point_b

    # x_coarse_b = x_a - offset[:, 0:1]
    # y_coarse_b = y_a - offset[:, 1:2]
    x_coarse_b = x_a - offset_a
    y_coarse_b = y_a - offset_b
    dis_x = abs(x_coarse_b - x_b)
    dis_y = abs(y_coarse_b - y_b)

    # new fashion
    # return torch.exp((dis_x * dis_x + dis_y * dis_y) / (-2.0 * 5))
    # old fashion
    return (one / (dis_x + one)) * (one / (dis_y + one))

class MatchScoreModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x_a, x_b, y_a, y_b, offset_a, offset_b, one):
        # x_a, y_a = point_a
        # x_b, y_b = point_b

        # x_coarse_b = x_a - offset[:, 0:1]
        # y_coarse_b = y_a - offset[:, 1:2]
        x_coarse_b = x_a - offset_a
        y_coarse_b = y_a - offset_b
        dis_x = abs(x_coarse_b - x_b)
        dis_y = abs(y_coarse_b - y_b)

        # new fashion
        # return torch.exp((dis_x * dis_x + dis_y * dis_y) / (-2.0 * 5))
        # old fashion
        return (one / (dis_x + one)) * (one / (dis_y + one))


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
    mlp_module = MatchScoreModule()
    # Create the module and compile it.
    compiled = compile_module(mlp_module)
    # compiled.dump()
    # Loads the compiled artifact into the runtime
    jit_module = BACKEND.load(compiled)
    print("jit_module:\n", jit_module.result)
    # Run it!
    result = jit_module.forward(torch.rand(1, 1).numpy(), 
                       torch.rand(1, 1).numpy(),
                       torch.rand(1, 1).numpy(),
                       torch.rand(1, 1).numpy(),
                       torch.rand(1, 1).numpy(),
                       torch.rand(1, 1).numpy(),
                       torch.Tensor([[1.]]).numpy(),)
    print("result: ", result)
