import torch
import torchvision

from functools import wraps

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import LinalgOnTensorsTosaBackend
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend


class ExternCallModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([800, 1333, 3], torch.float32, True),
        ([3], torch.float32, True),
        ([3], torch.float32, True),
    ])
    def forward(self, src, mean, scale):
        # RGB to BGR
        dup = src
        dup[:,:,0] = src[:,:,2]
        dup[:,:,2] = src[:,:,0]
        return (dup - mean) * scale


BACKEND = RefBackendLinalgOnTensorsBackend()
# BACKEND = LinalgOnTensorsTosaBackend()


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
    print("-------------------------------scripted._c-------------------------------")
    print(scripted._c)

    method = getattr(program, 'forward')
    args = []
    for arg in method._torch_mlir_arg_annotations:
        if not arg:
            continue
        assert arg[1] is torch.float32, "Only support float32"
        args.append(torch.rand(arg[0]))
    scripted2 = torch.jit.trace(program, args)

    ## Extract annotations.
    class_annotator = ClassAnnotator()
    extract_annotations(program, scripted, class_annotator)
    print("-------------------------------class_annotator---------------------------")
    print(class_annotator)


    ## Import the TorchScript module into MLIR.
    print("------------------Import TorchScript into MLIR-----------------------")
    mb = ModuleBuilder()
    mb.import_module(scripted._c, class_annotator)
    print("------------------printing-----init IR-------------------------------")
    mb.module.dump()
    print("------------------printing-----init IR------OVER---------------------")
    #return

    ## Lower the MLIR from TorchScript to RefBackend, passing through linalg-on-tensors.
    print("--------starting---------------torch-back-pipeline-------------------------------")
    pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline', mb.module.context)
    pm.run(mb.module)
    print("--------ending---------------torch-back-pipeline-------------------------------")
    mb.module.dump()
    print("--------printing---------------torch-back-pipeline----------OVER-----------------")

    pm = PassManager.parse('torch-backend-to-linalg-on-tensors-backend-pipeline', mb.module.context)
    print("----------starting-------------linalg----------PASS-----------------")
    pm.run(mb.module)
    print("----------ending---------------linalg----------PASS-----------------")
    mb.module.dump()
    print("----------printing-------------linalg----------OVER-----------------")
    return

    ## Invoke RefBackend to compile to compiled artifact form.
    compiled_module = BACKEND.compile(mb.module)
    print("-------------------------------after-compile-------------------------------")
    compiled_module.dump()
    return compiled_module


if __name__ == '__main__':
    print("---------------main-------------------")
    mlp_module = ExternCallModule()
    # Create the module and compile it.
    compiled = compile_module(mlp_module)
    # Loads the compiled artifact into the runtime
    jit_module = BACKEND.load(compiled)
    # Run it!
    x = torch.rand(800, 1333, 3).numpy()
    y = torch.rand(3).numpy()
    z = torch.rand(3).numpy()
    print("x: ", x, "\ny: ", y, "\nz: ", z)
    print("\n\n result: ", jit_module.forward(x, y, z))