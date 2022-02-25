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
    for (Operation *user : copy.getResult().getUsers()) {
      // We can only analyze within a single basic block.
      if (user->getBlock() != copy->getBlock())
        return failure();
      // We can only analyze these ops.
      if (!isa<CopyToValueTensorOp, OverwriteTensorOp>(user))
        return failure();
      users.push_back(user);
    }
    // Sort by order in the block, so we can abstractly interpret the ops.
    llvm::sort(users, [](Operation *lhs, Operation *rhs) {
      return lhs->isBeforeInBlock(rhs);
    });
    // Do an abstract interpretation within the block.
    // We track the current value tensor that holds the same contents as the
    // non-value tensor at each program point as we walk forward.
    Value currentlyHeldValueTensor = copy.getOperand();
    for (Operation *user : users) {
      if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(user)) {
        rewriter.replaceOp(copyToValueTensor, {currentlyHeldValueTensor});
      } else if (auto overwriteTensor = dyn_cast<OverwriteTensorOp>(user)) {
        currentlyHeldValueTensor = overwriteTensor.value();
        rewriter.eraseOp(overwriteTensor);
      } else {
        llvm_unreachable("only those ops supported!");
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
    // Find a subgraph starting with this CopyToNonValueTensorOp, and
    // terminating at CopyToValueTensorOp's, possibly with intervening view-like
    // ops.
    // This also catches the special case of a CopyToNonValueTensorOp that
    // trivially feeds into CopyToValueTensorOp's.
    SmallVector<Operation *> viewLikeOps;
    SmallVector<CopyToValueTensorOp> copyToValueTensorOps;
    auto workList = llvm::to_vector<6>(copy.getResult().getUsers());
    // We currently only support view-like ops with one tensor input and one
    // tensor output, meaning that the tensor use-def chains form a tree.
    // This will not be the case for an op like `torch.aten.view_as`, so
    // we will need to add a set to prune duplicate visitation.
    while (!workList.empty()) {
      Operation *op = workList.pop_back_val();
      if (auto copyToValueTensor = dyn_cast<CopyToValueTensorOp>(op)) {
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
        viewLikeOps.push_back(op);
        llvm::append_range(workList, op->getResult(0).getUsers());
      } else {
        return rewriter.notifyMatchFailure(
            copy, "can only handle these transitive user ops");
      }
    }

    copy.replaceAllUsesWith(copy.getOperand());
    for (CopyToValueTensorOp op : copyToValueTensorOps)
      rewriter.replaceOp(op, op.getOperand());
    for (Operation *op : viewLikeOps) {
      rewriter.updateRootInPlace(op, [&]() {
        if (auto nonValueTensorType =
                op->getResult(0).getType().dyn_cast<NonValueTensorType>()) {
          op->getResult(0).setType(nonValueTensorType.getWithValueSemantics());
        }
      });
    }
    return success();
  }
};
} // namespace

namespace {
// rewrite If op and If.yield op
class RewriteIfOp : public OpRewritePattern<PrimIfOp> {
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

      // todo: it might be a dfs within a tree
      auto nextUser = user->getResult(0).getUsers();
      llvm::append_range(workList, nextUser);
      llvm::errs() << "0 new users:\n";
      for (Operation *u : workList)
        u->dump();

      rankedVtensor = dyn_cast<CopyToValueTensorOp>(user);
    }

    // get ranked vtensor type
    auto type = rankedVtensor.getType();
    llvm::errs() << "0 ranked vtensor type:\t";
    type.dump();

    // rewrite ifOp's type with ranked vtensor
    ifOp.getResult(0).setType(type);

    // replace the CopyToValueTensorOp with PrimIfOp
    rewriter.replaceOp(rankedVtensor, ifOp.getResults());

//    // erase other ops in this path
//    for (auto op : workPath) {
//      rewriter.eraseOp(op);
//    }

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

//    // rewrite yieldOp's type with ranked vtensor
//    yieldOp.getResult(0).setType(yieldType);
	  llvm::errs() << "\n0 yield op:\t";
	  yieldOp.dump();

    // replace the CopyTOTensorOp with IfYieldOp
    rewriter.replaceOp(workPath.front(), HeldValueOp->getResult(0));
    workPath.erase(workPath.begin());

//    // erase other ops in this path
//    for (auto op : workPath) {
//      rewriter.eraseOp(op);
//    }
    return success();
  }
};
} // namespace

namespace {
// rewrite Loop op and Loop.condition op
class RewriteLoopOp : public OpRewritePattern<PrimLoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(PrimLoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    llvm::errs() << "----------1----------\n";
	  llvm::errs() << "1 before RewriteLoopOp:\n";
	  Block * currentBlock = rewriter.getBlock();
	  currentBlock->dump();

    // rewrite the PrimLoopOp
    CopyToValueTensorOp rankedVtensor;
    auto loopOpUsers = llvm::to_vector<6>(loopOp.results().getUsers());
    SmallVector<Operation *> workList;
	  SmallVector<Operation *> workPath;
	  workList.push_back(loopOpUsers.pop_back_val());
	  // todo: we only concern the first user in this path currently
	  while (!rankedVtensor && !workList.empty()) {
	  	Operation *user = workList.pop_back_val();
	  	if (!user)
	  		return failure();
	  	if (isa<mlir::ReturnOp, PrimLoopConditionOp>(user))
	  		return failure();

	  	workPath.push_back(user);
	  	llvm::errs() << "1 user in path:\t";
	  	user->dump();

	  	// todo: it might be a dfs within a tree
	  	auto nextUser = user->getResult(0).getUsers();
	  	llvm::append_range(workList, nextUser);
	  	llvm::errs() << "1 new users:\n";
	  	for (Operation *u : workList)
	  		u->dump();

	  	rankedVtensor = dyn_cast<CopyToValueTensorOp>(user);
	  }

	  // get ranked vtensor type
	  auto type = rankedVtensor.getType();
	  llvm::errs() << "1 ranked vtensor type:\t";
	  type.dump();

	  // rewrite loopOp's type with ranked vtensor
	  loopOp.getResult(0).setType(type);

	  // replace the CopyToValueTensorOp with PrimIfOp
	  rewriter.replaceOp(rankedVtensor, loopOp.getResults());

	  workPath.pop_back();
	  // erase other ops in this path
	  for (auto op : workPath) {
		  rewriter.eraseOp(op);
	  }

	  // rewrite the PrimLoopConditionOp
	  llvm::errs() << "\n1 last op in the region:\t";
	  Operation &loopConditionOp = loopOp.region().front().back();
	  loopConditionOp.dump();

	  // todo: we assume there's only one iter operand in the loopConditionOp
	  Operation *defOp = loopConditionOp.getOperand(1).getDefiningOp();
	  llvm::errs() << "1 loop condition def op\t";
	  defOp->dump();

	  workList.clear();
	  workPath.clear();
	  workList.push_back(defOp);
	  CopyToNonValueTensorOp rankedTensor;
	  while (!rankedTensor && ! workList.empty()) {
	  	Operation *definer = workList.pop_back_val();
	  	if (!definer)
	  		return failure();

	  	workPath.push_back(definer);
	  	llvm::errs() << "1 loop condition definer in path:\t";
	  	definer->dump();

	  	// todo: it will segment fault if the definer's operand comes from
      // function's operand
      auto previousDefiner = definer->getOperand(0).getDefiningOp();
      workList.push_back(previousDefiner);
      llvm::errs() << "1 loop condition previous definers:\n";
      for (Operation *u : workList) {
        u->dump();
      }

      rankedTensor = dyn_cast<CopyToNonValueTensorOp>(definer);
	  }

	  // get the currently held value op
    llvm::errs() << "1 loop condition the nearest user is:\t";
    rankedTensor.dump();
    Operation *HeldValueOp = rankedTensor.getOperand().getDefiningOp();
    llvm::errs() << "1 loop condition the value handled op is:\t";
    HeldValueOp->dump();
    // get ranked vtensor type
    auto loopConditionType = HeldValueOp->getResult(0).getType();
    llvm::errs() << "1 loop condition ranked tensor type:\t";
    loopConditionType.dump();
    // todo: check if it's the same type with PrimIfOp

//    // rewrite loopConditionOp's type with ranked vtensor
//    loopConditionOp.getResult(1).setType(loopConditionType);
	  llvm::errs() << "\n1 loop condition op:\t";
	  loopConditionOp.dump();

    // replace the CopyToTensorOp with loopConditionOp
    rewriter.replaceOp(workPath.front(), HeldValueOp->getResult(0));
//    workPath.erase(workPath.begin());

//    // erase other ops in this path
//    for (auto op : workPath) {
//      rewriter.eraseOp(op);
//    }

    // replace the primLoopOp
	  llvm::errs() << "loop arg:\t";
	  Value loopArg = loopOp.getOperand(2);
	  defOp = loopArg.getDefiningOp();
	  defOp->dump();
	  workList.clear();
	  workList.push_back(defOp);
	  CopyToNonValueTensorOp rankedTensorArg;
	  while (!rankedTensorArg && ! workList.empty()) {
		  Operation *definer = workList.pop_back_val();
		  if (!definer)
			  return failure();

		  llvm::errs() << "1 loop condition definer in path:\t";
		  definer->dump();

		  // todo: it will segment fault if the definer's operand comes from
		  // function's operand
		  auto previousDefiner = definer->getOperand(0).getDefiningOp();
		  workList.push_back(previousDefiner);
		  llvm::errs() << "1 loop condition previous definers:\n";
		  for (Operation *u : workList) {
			  u->dump();
		  }

		  rankedTensorArg = dyn_cast<CopyToNonValueTensorOp>(definer);
	  }

	  // get the currently held value op
	  llvm::errs() << "1 loop condition the nearest user is:\t";
	  rankedTensorArg.dump();
	  HeldValueOp = rankedTensorArg.getOperand().getDefiningOp();
	  llvm::errs() << "1 loop condition the value handled op is:\t";
	  HeldValueOp->dump();

	  // replace the CopyToTensorOp with loopConditionOp
	  rewriter.replaceOp(defOp, HeldValueOp->getResult(0));


	  llvm::errs() << "---------------------------------------------\n";
	  currentBlock = rewriter.getBlock();
	  currentBlock->dump();


	  // replace bb0's arg
	  llvm::errs() << "loopOp:\n";
	  loopOp.dump();
	  llvm::errs() << "1 bb0 arg:\t";
	  workList.clear();
	  loopOp.getRegion().front().dump();
	  BlockArgument bb0Arg = loopOp.getRegion().front().getArgument(1);
	  bb0Arg.dump();
	  auto bb0ArgUser = bb0Arg.getUsers();
	  llvm::append_range(workList, bb0ArgUser);
	  llvm::errs() << "1 bb0 arg users:\n";
	  for (Operation *u : workList)
		  u->dump();
	  CopyToValueTensorOp rankedTensorArgUser;
	  while (!rankedTensorArgUser && ! workList.empty()) {
		  Operation *user = workList.pop_back_val();
		  if (!user)
			  return failure();
		  if (isa<mlir::ReturnOp, PrimLoopConditionOp>(user))
			  return failure();

		  llvm::errs() << "1 user in path:\t";
		  user->dump();

		  // todo: it might be a dfs within a tree
		  auto nextUser = user->getResult(0).getUsers();
		  llvm::append_range(workList, nextUser);
		  llvm::errs() << "1 new users:\n";
		  for (Operation *u : workList)
			  u->dump();

		  rankedTensorArgUser = dyn_cast<CopyToValueTensorOp>(user);
	  }

	  llvm::errs() << "1 rankedTensorArgUser:\t";
	  rankedTensorArgUser->dump();

	  // get ranked vtensor type
	  type = rankedTensorArgUser.getType();
	  llvm::errs() << "1 ranked vtensor type:\t";
	  type.dump();

	  // rewrite bb0Arg's type with ranked vtensor
	  bb0Arg.setType(type);

	  // replace the CopyToValueTensorOp with bb0Arg
	  rewriter.replaceOp(rankedTensorArgUser, bb0Arg);


    llvm::errs() << "1 after RewriteLoopOp:\n";
    currentBlock = rewriter.getBlock();
    currentBlock->dump();
    return success();
  }
};
}

namespace {

class MaximizeValueSemanticsPass
    : public MaximizeValueSemanticsBase<MaximizeValueSemanticsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto func = getOperation();

    RewritePatternSet patterns(context);
    patterns.insert<RewriteIfOp, RewriteLoopOp,
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
