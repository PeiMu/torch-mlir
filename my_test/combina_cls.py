import torch
import torchvision

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.tosa_backends.linalg_on_tensors import LinalgOnTensorsTosaBackend
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend


class CombinaClsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Reset seed to make model deterministic.
        torch.manual_seed(0)
        self.mp = torch.nn.MaxPool2d(kernel_size=(1,2), stride=1)

    @export
    @annotate_args([
        None,
        ([1, 1, 32, 32], torch.float32, True),
        ([1, 1, 32, 32], torch.float32, True),
        ([1, 1, 32, 32], torch.float32, True),
        ([1, 1, 32, 32], torch.float32, True),
    ])
    def forward(self, tmp_No, tmp_ca, tmp_dr, tmp_fa):
        tmp_res = torch.cat([tmp_ca, tmp_fa], dim=0).permute(2,3,1,0)
        # if tmp_res.shape[2]!=1 or tmp_res.shape[3]!=1:
        tmp_res = self.mp(tmp_res)
        tmp_res = tmp_res.permute(3,2,0,1)
        hm_rel = torch.cat([tmp_No, tmp_res, tmp_dr], dim=0)
        hm_rel = hm_rel.permute(1,0,2,3)
        return hm_rel


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
    mlp_module = CombinaClsModule()
    # Create the module and compile it.
    compiled = compile_module(mlp_module)
    # Loads the compiled artifact into the runtime
    jit_module = BACKEND.load(compiled)
    # Run it!
    tmp_No = torch.rand(1, 1, 32, 32).numpy()
    tmp_ca = torch.rand(1, 1, 32, 32).numpy()
    tmp_dr = torch.rand(1, 1, 32, 32).numpy()
    tmp_fa = torch.rand(1, 1, 32, 32).numpy()
    print("\n\n result: ", jit_module.forward(tmp_No, tmp_ca, tmp_dr, tmp_fa))
