//===- MaximizeValueSemantics.cpp --------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;

namespace {
class AbstractlyInterpretCopyToNonValueTensorOpUsersWithinABlock
    : public OpRewritePattern<CopyToNonValueTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyToNonValueTensorOp copy,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> users;
    // See if our limited form of analysis is even applicatble.
    llvm::errs() << "----------1----------\n";
    llvm::errs() << "1 copy:\t";
    copy->dump();
    for (Operation *user : copy.getResult().getUsers()) {
      // We can only analyze within a single basic block.
      llvm::errs() << "1 user:\t";
      user->dump();
      if (user->getBlock() != copy->getBlock())
        return failure();
      // We can only analyze these ops.

      if (!isa<CopyToValueTensorOp, OverwriteTensorOp>(user)) {
        llvm::errs() << "1 do not process here\n";
        return failure();
      }
      users.push_back(user);
    }
    llvm::errs() << "1 users:\n";
    for (Operation *u : users) {
      u->dump();
    }
    // Sort by order in the block, so we can abstractly interpret the ops.
    llvm::sort(users, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    llvm::errs() << "1 after sort users:\n";
    for (Operation *u : users) {
      u->dump();
    }
    // Do an abstract interpretation within the block.
    // We track the current value tensor that holds the same contents as the
    // non-value tensor at each program point as we walk forward.
    Value currentlyHeldValueTensor = copy.getOperand();
    llvm::errs() << "1 currentlyHeldValueTensor\t";
    currentlyHeldValueTensor.dump();
    for (Operation *user : users) {
      if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(user)) {
        llvm::errs() << "1 replace copy to vtensor\t";
        copyToValueTensor->dump();
        rewriter.replaceOp(copyToValueTensor, {currentlyHeldValueTensor});
      } else if (auto overwriteTensor = dyn_cast<OverwriteTensorOp>(user)) {
        llvm::errs() << "1 replace copy to vtensor\t";
        overwriteTensor->dump();
        currentlyHeldValueTensor = overwriteTensor.value();
        rewriter.eraseOp(overwriteTensor);
      } else {
        llvm::errs() << "1 do not support:\n";
        user->dump();
        llvm_unreachable("1 only those ops supported!");
      }
    }
    rewriter.eraseOp(copy);
    return success();
  }
};
} // namespace

namespace {
// Calculate a forward slice starting from a CopyToNonValueTensorOp
// and ending at CopyToValueTensorOp's. If all intervening ops
// are just view-like operations (i.e. no mutation), then we can trivially
// convert them all to value semantics.
class RewriteViewLikeSubgraph
    : public OpRewritePattern<CopyToNonValueTensorOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(CopyToNonValueTensorOp copy,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "----------2----------\n";
    llvm::errs() << "2 copy:\t";
    copy->dump();
    // Find a subgraph starting with this CopyToNonValueTensorOp, and
    // terminating at CopyToValueTensorOp's, possibly with intervening view-like
    // ops.
    // This also catches the special case of a CopyToNonValueTensorOp that
    // trivially feeds into CopyToValueTensorOp's.
    SmallVector<Operation *> viewLikeOps;
    SmallVector<CopyToValueTensorOp> copyToValueTensorOps;
    auto workList = llvm::to_vector<6>(copy.getResult().getUsers());
    llvm::errs() << "2 users:\n";
    for (Operation *u : workList) {
      u->dump();
    }
    // We currently only support view-like ops with one tensor input and one
    // tensor output, meaning that the tensor use-def chains form a tree.
    // This will not be the case for an op like `torch.aten.view_as`, so
    // we will need to add a set to prune duplicate visitation.
    while (!workList.empty()) {
      Operation *op = workList.pop_back_val();
      llvm::errs() << "2 current user:\t";
      op->dump();
      if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(op)) {
        llvm::errs() << "----------2 insert to the C2Vtensor set----------\n";
        copyToValueTensorOps.push_back(copyToValueTensor);
      } else if (isa<AtenSqueezeOp, AtenSqueezeDimOp, AtenUnsqueezeOp,
                     AtenFlattenUsingIntsOp, AtenTransposeIntOp,
                     TensorStaticInfoCastOp, AtenBroadcastToOp, AtenToDtypeOp,
                     AtenContiguousOp, AtenPermuteOp, AtenViewOp, AtenExpandOp,
                     AtenFill_ScalarOp, AtenSliceTensorOp, AtenSelectIntOp,
                     AtenTOp>(op)) {
        // AtenContiguousOp might return a view, so this is conservatively
        // correct. We could potentially be more precise and identify the cases
        // that it does not return a view and treat those as having value
        // semantics.
        llvm::errs() << "----------2 insert to the view set----------\n";
        viewLikeOps.push_back(op);
        llvm::errs()
            << "----------2 refresh users, add view_op's user----------\n";
        llvm::append_range(workList, op->getResult(0).getUsers());
        llvm::errs() << "2 new users:\n";
        for (Operation *u : workList) {
          u->dump();
        }
      } else {
        llvm::errs() << "2 do not support:\n";
        return rewriter.notifyMatchFailure(
            copy, "can only handle these transitive user ops");
      }
    }
    llvm::errs() << "2 C2Vtensor set:\n";
    for (CopyToValueTensorOp u : copyToValueTensorOps) {
      u.dump();
    }
    llvm::errs() << "2 view set:\n";
    for (Operation *u : viewLikeOps) {
      u->dump();
    }

    llvm::errs() << "2 copy's operand:\t";
    copy.getOperand().dump();
    copy.replaceAllUsesWith(copy.getOperand());
    llvm::errs() << "2 copy after replace all uses:\t";
    copy.dump();
    for (CopyToValueTensorOp op : copyToValueTensorOps) {
      llvm::errs() << "2 C2Vtensor:\t";
      op.dump();
      llvm::errs() << "2 replace with:\t";
      op.getOperand().dump();
      rewriter.replaceOp(op, op.getOperand());
    }
    for (Operation *op : viewLikeOps) {
      llvm::errs() << "2 old view op:\t";
      op->dump();
      rewriter.updateRootInPlace(op, [&]() {
        if (auto nonValueTensorType =
                op->getResult(0).getType().dyn_cast<NonValueTensorType>()) {
          op->getResult(0).setType(nonValueTensorType.getWithValueSemantics());
        }
      });
      llvm::errs() << "2 new view op:\t";
      op->dump();
    }
    return success();
  }
};
} // namespace

namespace {
// rewrite If op and If.yield op
class RewriteIfLikeOp : public OpRewritePattern<PrimIfOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(PrimIfOp ifOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "----------0----------\n";
    llvm::errs() << "0 ifOp:\n";
    ifOp->dump();

    // rewrite the PrimIfOp
    CopyToValueTensorOp rankedVtensor;
    auto ifOpUsers = llvm::to_vector<6>(ifOp.results().getUsers());
    SmallVector<Operation *> workList;
    SmallVector<Operation *> workPath;
    workList.push_back(ifOpUsers.pop_back_val());
    // todo: we only concern the first user in this path currently
    while (!rankedVtensor && !workList.empty()) {
      Operation *user = workList.pop_back_val();
      if (!user)
        return failure();
      if (isa<mlir::ReturnOp, PrimIfYieldOp>(user))
        return failure();

      workPath.push_back(user);
      llvm::errs() << "0 user in path:\t";
      user->dump();

      // todo: it might be a dfs with a tree
      auto nextUser = user->getResult(0).getUsers();
      llvm::append_range(workList, nextUser);
      llvm::errs() << "0 new users:\n";
      for (Operation *u : workList) {
        u->dump();
      }

      rankedVtensor = dyn_cast<CopyToValueTensorOp>(user);
    }

    // get ranked vtensor type
    auto type =
        dyn_cast<CopyToValueTensorOp>(workPath.pop_back_val()).getType();
    llvm::errs() << "0 ranked vtensor type:\t";
    type.dump();

    // rewrite ifOp's type with ranked vtensor
    ifOp.getResult(0).setType(type);

    // replace the CopyToValueTensorOp with PrimIfOp
    rewriter.replaceOp(rankedVtensor, ifOp.getResults());

    // erase other ops in this path
    for (auto op : workPath) {
      rewriter.eraseOp(op);
    }

    // rewrite the PrimIfYieldOp
    // in the then region
    llvm::errs() << "\n0 last op in the then region:\t";
    Operation &thenYieldOp = ifOp.thenRegion().front().back();
    thenYieldOp.dump();
    if (failed(RewriteControlRegion(thenYieldOp, rewriter)))
      return failure();

    // in the else region
    llvm::errs() << "\n0 last op in the else region:\t";
    Operation &elseYieldOp = ifOp.elseRegion().front().back();
    elseYieldOp.dump();
    if (failed(RewriteControlRegion(elseYieldOp, rewriter)))
      return failure();

    return success();
  }

  LogicalResult RewriteControlRegion(Operation &yieldOp,
                                     PatternRewriter &rewriter) const {
    // todo: we assume there's only one operand in the yieldOp
    Operation *defOp = yieldOp.getOperand(0).getDefiningOp();
    llvm::errs() << "0 yield def op:\t";
    defOp->dump();

    SmallVector<Operation *> workList;
    SmallVector<Operation *> workPath;
    workList.push_back(defOp);
    CopyToNonValueTensorOp rankedTensor;
    while (!rankedTensor && !workList.empty()) {
      Operation *definer = workList.pop_back_val();
      if (!definer)
        return failure();

      workPath.push_back(definer);
      llvm::errs() << "0 yield definer in path:\t";
      definer->dump();

      // todo: it will segment fault if the definer's operand comes from
      // function's operand
      auto previousDefiner = definer->getOperand(0).getDefiningOp();
      workList.push_back(previousDefiner);
      llvm::errs() << "0 yield previous definers:\n";
      for (Operation *u : workList) {
        u->dump();
      }

      rankedTensor = dyn_cast<CopyToNonValueTensorOp>(definer);
    }

    // get the currently held value op
    llvm::errs() << "0 yield the nearest user is:\t";
    rankedTensor.dump();
    Operation *HeldValueOp = rankedTensor.getOperand().getDefiningOp();
    llvm::errs() << "0 yield the value handled op is:\t";
    HeldValueOp->dump();
    // get ranked vtensor type
    auto yieldType = HeldValueOp->getResult(0).getType();
    llvm::errs() << "0 yield ranked tensor type:\t";
    yieldType.dump();
    // todo: check if it's the same type with PrimIfOp

    // rewrite yieldOp's type with ranked vtensor
//    yieldOp.getResult(0).setType(yieldType);
	  llvm::errs() << "\n0 yield op:\t";
	  yieldOp.dump();

    // replace the CopyTOTensorOp with IfYieldOp
    rewriter.replaceOp(workPath.front(), HeldValueOp->getResult(0));
    workPath.erase(workPath.begin());

    // erase other ops in this path
    for (auto op : workPath) {
      rewriter.eraseOp(op);
    }
    return success();
  }
};
} // namespace

namespace {

class MaximizeValueSemanticsPass
    : public MaximizeValueSemanticsBase<MaximizeValueSemanticsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();

    RewritePatternSet patterns(context);
    patterns.insert<RewriteIfLikeOp,
                    AbstractlyInterpretCopyToNonValueTensorOpUsersWithinABlock,
                    RewriteViewLikeSubgraph>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::torch::Torch::createMaximizeValueSemanticsPass() {
  return std::make_unique<MaximizeValueSemanticsPass>();
}
