import torch
import torchvision

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import LinalgOnTensorsTosaBackend


class Mlp1LayerModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.fc0 = torch.nn.Linear(3, 5)
        self.tanh0 = torch.nn.Tanh()

    @export
    @annotate_args([
        None,
        ([-1, -1], torch.float32, True),
    ])
    def forward(self, x):
        return self.tanh0(self.fc0(x))


BACKEND = LinalgOnTensorsTosaBackend()

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
    mlp_module = Mlp1LayerModule()
    # Create the module and compile it.
    compiled = compile_module(mlp_module)
    # compiled.dump()
    # Loads the compiled artifact into the runtime
    jit_module = BACKEND.load(compiled)
    # Run it!
    jit_module.forward(torch.rand(5, 3).numpy())
